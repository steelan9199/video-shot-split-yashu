#!/usr/bin/env python
"""按镜头切分视频。

主检：TransNetV2（逐帧转场概率）。
复核：PySceneDetect ContentDetector，只对 0.3–0.5 的不确定带做 yes/no 裁决，
      不产生新切点，不参与并集。
后处理：1 秒内命中合并 → 单镜上限（默认 14 秒）强制均分。
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="按镜头切分视频（TransNetV2 主检 + PySceneDetect 裁决）")
    p.add_argument("video", help="输入视频绝对路径")
    p.add_argument("--outdir", help="输出目录，默认 <视频名>_shots")
    p.add_argument("--max-sec", type=float, default=14.0, help="单镜最长秒数，0 表示不限制")
    p.add_argument("--min-sec", type=float, default=0.5, help="单镜最短秒数，更短的并入相邻段")
    p.add_argument("--merge-window", type=float, default=1.0, help="切点合并窗口（秒）")
    p.add_argument("--hi", type=float, default=0.5, help="TransNetV2 直接采信阈值")
    p.add_argument("--lo", type=float, default=0.3, help="不确定带下界（低于此值直接丢弃）")
    p.add_argument("--adjudicate-tol", type=float, default=0.5, help="PySceneDetect 裁决容差（秒）")
    p.add_argument("--no-scenedetect", action="store_true", help="关闭复核，只用 TransNetV2")
    p.add_argument("--copy", action="store_true",
                   help="流复制切片（快，但边界会吸附到关键帧；默认重编码保证帧级准确）")
    p.add_argument("--dry-run", action="store_true", help="只出切点表，不切文件")
    p.add_argument("--device", default="auto", help="auto / cuda / cpu")
    p.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg 可执行文件路径")
    p.add_argument("--ffprobe", default="ffprobe", help="ffprobe 可执行文件路径")
    return p.parse_args()


def probe(path, ffprobe):
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,r_frame_rate:format=duration",
           "-of", "json", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe 失败: {out.stderr.strip()}")
    data = json.loads(out.stdout)
    st = data["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    return {
        "width": int(st["width"]),
        "height": int(st["height"]),
        "fps": float(num) / float(den),
        "duration": float(data["format"]["duration"]),
    }


def transnet_probs(video, device):
    import torch
    from transnetv2_pytorch import TransNetV2

    model = TransNetV2(device=device)
    model.eval()
    res = model.analyze_video(str(video))
    fps = float(res["fps"])
    preds = res["single_frame_predictions"]
    if hasattr(preds, "detach"):
        preds = preds.detach().cpu().numpy()
    preds = np.asarray(preds, dtype=np.float32).reshape(-1)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return fps, preds


def peaks(preds, start, end, thr):
    """在 [start,end) 找连续超阈值的段，取段内最大值为一个切点。"""
    out = []
    i = start
    while i < end:
        if preds[i] >= thr:
            j = i
            while j + 1 < end and preds[j + 1] >= thr:
                j += 1
            k = i + int(np.argmax(preds[i:j + 1]))
            out.append((k, float(preds[k])))
            i = j + 1
        else:
            i += 1
    return out


def local_peaks(preds, lo, hi, fps, min_gap_sec=0.5):
    """不确定带内的局部峰：>=lo 且 <hi，左右 ±min_gap 内最大。"""
    w = max(1, int(round(min_gap_sec * fps)))
    n = len(preds)
    out = []
    for k in range(n):
        v = float(preds[k])
        if v < lo or v >= hi:
            continue
        left = preds[max(0, k - w):k]
        right = preds[k + 1:k + 1 + w]
        if (left.size == 0 or v >= float(left.max())) and (right.size == 0 or v >= float(right.max())):
            if out and (k - out[-1][0]) < w:
                if v > out[-1][1]:
                    out[-1] = (k, v)
                continue
            out.append((k, v))
    return out


def scenedetect_times(video):
    from scenedetect import detect, ContentDetector
    scenes = detect(str(video), ContentDetector())
    times = []
    for start, end in scenes[1:]:
        times.append(start.seconds)
    return times


def keyframes(video, ffprobe):
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
           "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(video)]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    vals = [x.strip().strip(",") for x in out.stdout.split()]
    return sorted(float(v) for v in vals if v)


def snap_to_keyframes(cuts, kfs):
    """流复制只能从关键帧起切，先把切点吸附到最近关键帧，避免整段回退造成重复。"""
    if not kfs:
        return cuts
    for c in cuts:
        c["time"] = min(kfs, key=lambda k: abs(k - c["time"]))
    cuts.sort(key=lambda c: c["time"])
    dedup = []
    for c in cuts:
        if dedup and abs(c["time"] - dedup[-1]["time"]) < 1e-3:
            continue
        dedup.append(c)
    return dedup


def build_cuts(preds, fps, sd_times, duration, args):
    n = len(preds)
    accepted = peaks(preds, 0, n, args.hi)
    cuts = [{"time": k / fps, "prob": v, "src": "transnetv2"} for k, v in accepted]

    uncertain_n = adjudicated_n = 0
    if sd_times is not None:
        uncertain = local_peaks(preds, args.lo, args.hi, fps)
        uncertain_n = len(uncertain)
        for k, v in uncertain:
            t = k / fps
            if not sd_times:
                continue
            d = min(abs(t - s) for s in sd_times)
            if d <= args.adjudicate_tol:
                adjudicated_n += 1
                cuts.append({"time": t, "prob": v, "src": f"transnetv2+scenedetect({d:.2f}s)"})

    # 片头片尾的黑场/淡入淡出常被判成 0.x 秒的假切点，直接丢弃
    cuts = [c for c in cuts if args.min_sec <= c["time"] <= duration - args.min_sec]

    cuts.sort(key=lambda c: c["time"])
    merged = []
    for c in cuts:
        if merged and c["time"] - merged[-1]["time"] < args.merge_window:
            if c["prob"] > merged[-1]["prob"]:
                merged[-1] = c
            continue
        merged.append(c)
    stats = {"hi_hits": len(accepted), "uncertain": uncertain_n,
             "adjudicated": adjudicated_n, "final": len(merged)}
    return merged, stats


def build_shots(cuts, duration, args):
    bounds = [0.0] + [c["time"] for c in cuts] + [duration]
    bounds = [b for b in sorted(bounds) if 0.0 <= b <= duration]
    shots = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if e - s <= 1e-3:
            continue
        if shots and (e - s) < args.min_sec:
            shots[-1]["end"] = e
            shots[-1]["duration"] = round(e - shots[-1]["start"], 3)
            continue
        shots.append({"index": len(shots) + 1, "start": round(s, 3), "end": round(e, 3),
                      "duration": round(e - s, 3), "forced": False})

    if args.max_sec and args.max_sec > 0:
        capped = []
        for sh in shots:
            d = sh["end"] - sh["start"]
            if d <= args.max_sec + 1e-6:
                capped.append(sh)
                continue
            parts = math.ceil(d / args.max_sec)
            step = d / parts
            for k in range(parts):
                s = sh["start"] + k * step
                e = sh["start"] + (k + 1) * step if k < parts - 1 else sh["end"]
                capped.append({"index": len(capped) + 1, "start": round(s, 3),
                               "end": round(e, 3), "duration": round(e - s, 3),
                               "forced": True})
        shots = capped

    for i, sh in enumerate(shots, 1):
        sh["index"] = i
    return shots


def _ffmpeg_cut(video, sh, dst, args, reencode):
    cmd = [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{sh['start']:.3f}", "-i", str(video), "-t", f"{sh['duration']:.3f}"]
    if reencode:
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    cmd.append(str(dst))
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def run_ffmpeg_split(video, shots, outdir, args):
    outdir.mkdir(parents=True, exist_ok=True)
    made = []
    for sh in shots:
        name = f"{sh['index']:03d}_{sh['start']:.3f}-{sh['end']:.3f}.mp4"
        dst = outdir / name
        r = _ffmpeg_cut(video, sh, dst, args, not args.copy)
        if r.returncode != 0:
            print(f"  [失败] {name}: {r.stderr.strip()[:300]}", file=sys.stderr)
            continue
        if args.copy and args.max_sec and args.max_sec > 0:
            d = probe(dst, args.ffprobe)["duration"]
            if d > args.max_sec + 0.2:
                r2 = _ffmpeg_cut(video, sh, dst, args, True)
                if r2.returncode == 0:
                    print(f"  [兜底重编码] {name}: copy {d:.2f}s → 上限内")
        made.append((dst, sh))
    return made


def _small(img):
    import cv2
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (96, 54)).astype(np.float32)


def _head_tail(path, n=12):
    """取片段前 n 帧和后 n 帧（缩小灰度），用于重复与边界校验。"""
    import cv2
    from collections import deque
    cap = cv2.VideoCapture(str(path))
    head = []
    tail = deque(maxlen=n)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        s = _small(f)
        tail.append(s)
        if len(head) < n:
            head.append(s)
    cap.release()
    if not head:
        return None, None
    return np.array(head), np.array(list(tail))


def _src_frame(video, t, fps):
    """按帧号取源片帧，避免 OpenCV 毫秒级 seek 落到关键帧。"""
    import cv2
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, f = cap.read()
    cap.release()
    return _small(f) if ok else None


def verify(made, video, fps, args):
    """时长 + 起点是否对准源片 + 是否整段重复。"""
    issues = []
    starts = []
    prev_head = prev_tail = None
    for path, sh in made:
        info = probe(path, args.ffprobe)
        if args.max_sec and info["duration"] > args.max_sec + 0.2:
            issues.append(f"{path.name} 时长 {info['duration']:.2f}s 超过上限")
        head, tail = _head_tail(path)
        if head is None:
            continue
        # 起点允许 ±1 帧：ffmpeg 取「不早于 -ss 的第一帧」，硬切处差一帧就是两个画面
        idx = int(round(sh["start"] * fps))
        cands = []
        for k in (idx - 1, idx, idx + 1):
            ref = _src_frame(video, k / fps, fps)
            if ref is not None:
                cands.append(float(np.abs(head[0] - ref).mean()))
        start_diff = min(cands) if cands else -1
        starts.append(start_diff)
        if start_diff > 12:
            issues.append(f"{path.name} 起点与源片 {sh['start']:.3f}s 不符（像素差 {start_diff:.1f}）")
        # 强制均分段本来就是同一镜头内部切开，相邻帧连续是正常的；只查真实切点
        if prev_head is not None and not sh.get("forced"):
            head_dup = float(np.abs(head - prev_head).mean())
            tail_dup = float(np.abs(head - prev_tail).mean())
            if head_dup < 2.0 or tail_dup < 2.0:
                issues.append(f"{path.name} 与上一段重复（首帧差 {head_dup:.2f} / 尾帧差 {tail_dup:.2f}）")
        prev_head, prev_tail = head, tail
    med = float(np.median([s for s in starts if s >= 0])) if starts else -1
    return issues, med


def main():
    args = parse_args()
    video = Path(args.video).expanduser().resolve()
    if not video.exists():
        sys.exit(f"找不到视频: {video}")
    outdir = Path(args.outdir) if args.outdir else video.parent / f"{video.stem}_shots"

    meta = probe(video, args.ffprobe)
    print(f"[1/4] 源片 {meta['width']}x{meta['height']} @ {meta['fps']:.3f}fps, {meta['duration']:.3f}s")

    fps, preds = transnet_probs(video, args.device)
    print(f"[2/4] TransNetV2 完成：{len(preds)} 帧，fps={fps:.3f}")

    sd_times = None
    if not args.no_scenedetect:
        try:
            sd_times = scenedetect_times(video)
            print(f"       PySceneDetect 参考切点 {len(sd_times)} 个")
        except Exception as exc:
            print(f"       PySceneDetect 不可用，跳过复核: {exc}", file=sys.stderr)

    cuts, stats = build_cuts(preds, fps, sd_times, meta["duration"], args)
    if args.copy:
        kfs = keyframes(video, args.ffprobe)
        cuts = snap_to_keyframes(cuts, kfs)
        stats["snapped"] = True
        print(f"       流复制模式：切点已吸附到最近关键帧（共 {len(kfs)} 个关键帧）")
    shots = build_shots(cuts, meta["duration"], args)
    forced = sum(1 for s in shots if s.get("forced"))
    print(f"[3/4] 硬切命中 {stats['hi_hits']} → 不确定带候选 {stats['uncertain']}"
          f" → 复核采纳 {stats['adjudicated']} → 合并后切点 {stats['final']}"
          f" → 镜头段 {len(shots)} 个（其中 14 秒上限均分 {forced} 段）")

    outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "source": str(video),
        "video": meta,
        "params": {"hi": args.hi, "lo": args.lo, "merge_window": args.merge_window,
                   "adjudicate_tol": args.adjudicate_tol, "max_sec": args.max_sec,
                   "min_sec": args.min_sec, "scenedetect": sd_times is not None},
        "cuts": cuts,
        "counts": stats,
        "shots": shots,
    }
    (outdir / "shots.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(outdir / "shots.csv", "w", encoding="utf-8-sig", newline="") as fh:
        fh.write("index,start,end,duration,boundary\n")
        for sh in shots:
            kind = "forced" if sh.get("forced") else "shot"
            fh.write(f"{sh['index']},{sh['start']:.3f},{sh['end']:.3f},{sh['duration']:.3f},{kind}\n")
    print(f"       切点表: {outdir / 'shots.json'}")

    if args.dry_run:
        return
    made = run_ffmpeg_split(video, shots, outdir, args)
    print(f"[4/4] 已切 {len(made)} 个文件 → {outdir}")
    issues, med = verify(made, video, meta["fps"], args)
    print(f"       边界校验：起点与源片像素差中位数 {med:.2f}（<12 为对准）")
    if issues:
        print("       [需人工确认]", file=sys.stderr)
        for it in issues:
            print(f"       - {it}", file=sys.stderr)
    else:
        print("       验收通过：时长全部达标，无重复片段，起点对准源片")


if __name__ == "__main__":
    main()
