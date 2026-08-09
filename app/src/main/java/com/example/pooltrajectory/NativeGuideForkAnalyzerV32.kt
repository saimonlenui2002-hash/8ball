package com.example.pooltrajectory

import android.content.Context
import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.max
import kotlin.math.min

/**
 * v3.2 is tuned for the wide 8 Ball Pool layout used on the target device.
 *
 * Important differences from v3.1:
 * - the table ROI is stable in normalized screen coordinates and only validated by felt colour;
 * - cue-ball endpoints are penalized by detecting a nearby solid white patch;
 * - random line endpoints are rejected unless enough native white guide pixels exist locally;
 * - extensions start exactly at the native collision/fork junction.
 */
class NativeGuideForkAnalyzerV32(private val context: Context) {
    private val prefs = context.getSharedPreferences(MainActivity.PREFS, Context.MODE_PRIVATE)

    private data class GuideSeg(val p1: Vec2, val p2: Vec2, val length: Double)
    private data class Branch(val direction: Vec2, val length: Double, val quality: Double)
    private data class Fork(
        val junction: Vec2,
        val incoming: Vec2,
        val incomingSeg: GuideSeg,
        val branches: List<Branch>,
        val solidWhite: Double,
        val localWhite: Double
    )

    private data class WhiteIntegral(
        val width: Int,
        val height: Int,
        val stride: Int,
        val data: IntArray
    )

    fun analyze(bitmap: Bitmap): AnalysisResult {
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)
        val scale = min(1.0, 1280.0 / src.cols().toDouble())
        val work = Mat()
        if (scale < 0.999) Imgproc.resize(src, work, Size(), scale, scale, Imgproc.INTER_AREA)
        else src.copyTo(work)
        src.release()

        val tableRect = fixedValidatedTable(work)
        if (tableRect == null) {
            work.release()
            return AnalysisResult(
                bitmap.width, bitmap.height, null, emptyList(), null,
                emptyList(), null, 0, "стол не подтвержден"
            )
        }

        val table = work.submat(tableRect)
        val segments = detectGuideSegments(table)
        val whiteIntegral = buildWhiteIntegral(table)
        val fork = findFork(segments, table.cols().toDouble(), whiteIntegral)

        val playRect = PlayRect(
            tableRect.x / scale,
            tableRect.y / scale,
            (tableRect.x + tableRect.width) / scale,
            (tableRect.y + tableRect.height) / scale
        )

        val out = mutableListOf<TrajectorySegment>()
        val debug = prefs.getBoolean(MainActivity.KEY_DEBUG, false)

        if (fork != null) {
            val ordered = fork.branches.sortedBy { branchAngle(fork.incoming, it.direction) }
            ordered.take(2).forEachIndexed { index, branch ->
                val kind = if (index == 0) SegmentKind.OBJECT else SegmentKind.CUE_AFTER
                val bounces = if (kind == SegmentKind.OBJECT) {
                    prefs.getInt(MainActivity.KEY_BOUNCES, 0).coerceIn(0, 2)
                } else 0

                // Start exactly at the detected native contact marker. This deliberately
                // overlaps the game's short native branch so there is no visible offset.
                out += extendBranch(
                    fork.junction,
                    branch.direction,
                    tableRect,
                    scale,
                    kind,
                    bounces
                )
            }

            if (debug) {
                out += TrajectorySegment(
                    toGlobal(fork.incomingSeg.p1, tableRect, scale),
                    toGlobal(fork.incomingSeg.p2, tableRect, scale),
                    SegmentKind.AIM
                )
                val j = fork.junction
                val cross = max(7.0, tableRect.width * 0.009)
                out += TrajectorySegment(
                    toGlobal(Vec2(j.x - cross, j.y), tableRect, scale),
                    toGlobal(Vec2(j.x + cross, j.y), tableRect, scale),
                    SegmentKind.AIM
                )
                out += TrajectorySegment(
                    toGlobal(Vec2(j.x, j.y - cross), tableRect, scale),
                    toGlobal(Vec2(j.x, j.y + cross), tableRect, scale),
                    SegmentKind.AIM
                )
            }
        } else if (debug) {
            segments.take(14).forEach { s ->
                out += TrajectorySegment(
                    toGlobal(s.p1, tableRect, scale),
                    toGlobal(s.p2, tableRect, scale),
                    SegmentKind.AIM
                )
            }
        }

        table.release()
        work.release()

        val status = when {
            segments.isEmpty() -> "штатная линия не найдена"
            fork == null -> "контактный узел не найден"
            fork.branches.size == 1 -> "узел найден • 1 ветка"
            else -> "узел найден • 2 ветки"
        }
        val confidence = when {
            segments.isEmpty() -> 25
            fork == null -> 45
            fork.branches.size == 1 -> 86
            else -> 98
        }

        return AnalysisResult(
            bitmap.width, bitmap.height, null, emptyList(), null,
            out, playRect, confidence, status
        )
    }

    /**
     * The game table has a stable position on the target phone. Using the normalized ROI
     * removes the intermittent contour failure seen in v3.1. We still validate the ROI
     * so loading/menu screens never receive trajectories.
     */
    private fun fixedValidatedTable(work: Mat): Rect? {
        if (work.cols() < 300 || work.rows() < 200) return null

        val x = (work.cols() * 0.1787).toInt().coerceIn(0, work.cols() - 2)
        val y = (work.rows() * 0.2080).toInt().coerceIn(0, work.rows() - 2)
        val right = (work.cols() * 0.8215).toInt().coerceIn(x + 2, work.cols())
        val bottom = (work.rows() * 0.9410).toInt().coerceIn(y + 2, work.rows())
        val rect = Rect(x, y, right - x, bottom - y)

        val rgb = Mat()
        val hsv = Mat()
        Imgproc.cvtColor(work, rgb, Imgproc.COLOR_RGBA2RGB)
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
        rgb.release()

        val hist = IntArray(180)
        var valid = 0
        val x0 = rect.x + (rect.width * 0.08).toInt()
        val x1 = rect.x + (rect.width * 0.92).toInt()
        val y0 = rect.y + (rect.height * 0.08).toInt()
        val y1 = rect.y + (rect.height * 0.92).toInt()
        val step = max(4, rect.width / 150)

        var yy = y0
        while (yy < y1) {
            var xx = x0
            while (xx < x1) {
                val p = hsv.get(yy, xx)
                if (p != null && p.size >= 3 && p[1] >= 45.0 && p[2] >= 45.0) {
                    hist[p[0].toInt().coerceIn(0, 179)]++
                    valid++
                }
                xx += step
            }
            yy += step
        }

        if (valid < 180) {
            hsv.release()
            return null
        }
        val dominant = hist.indices.maxByOrNull { hist[it] } ?: run {
            hsv.release(); return null
        }

        var cluster = 0
        for (d in -11..11) {
            var h = dominant + d
            if (h < 0) h += 180
            if (h >= 180) h -= 180
            cluster += hist[h]
        }
        hsv.release()

        val share = cluster.toDouble() / valid.toDouble()
        // Blue/green pool cloth plus a strong dominant hue. This rejects the black loader.
        return if (dominant in 35..125 && share >= 0.42) rect else null
    }

    private fun detectGuideSegments(tableRgba: Mat): List<GuideSeg> {
        val rgb = Mat()
        val hsv = Mat()
        val mask = Mat()
        Imgproc.cvtColor(tableRgba, rgb, Imgproc.COLOR_RGBA2RGB)
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
        rgb.release()

        // Native 8BP guide is bright and nearly achromatic.
        Core.inRange(hsv, Scalar(0.0, 0.0, 150.0), Scalar(179.0, 82.0, 255.0), mask)
        hsv.release()

        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0, 2.0))
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel)
        kernel.release()

        val border = max(5, (tableRgba.cols() * 0.008).toInt())
        Imgproc.rectangle(
            mask,
            Point(0.0, 0.0),
            Point((mask.cols() - 1).toDouble(), (mask.rows() - 1).toDouble()),
            Scalar(0.0),
            border
        )

        val lines = Mat()
        val minLen = max(18.0, tableRgba.cols() * 0.020)
        Imgproc.HoughLinesP(mask, lines, 1.0, Math.PI / 360.0, 17, minLen, 12.0)

        val out = mutableListOf<GuideSeg>()
        for (row in 0 until lines.rows()) {
            val l = lines.get(row, 0) ?: continue
            if (l.size < 4) continue
            val p1 = Vec2(l[0], l[1])
            val p2 = Vec2(l[2], l[3])
            val len = (p2 - p1).length()
            if (len < minLen) continue

            val mx = (p1.x + p2.x) * 0.5
            val my = (p1.y + p2.y) * 0.5
            val px = tableRgba.cols() * 0.014
            val py = tableRgba.rows() * 0.018
            if (mx < px || mx > tableRgba.cols() - px || my < py || my > tableRgba.rows() - py) continue
            out += GuideSeg(p1, p2, len)
        }

        lines.release()
        mask.release()
        return out.sortedByDescending { it.length }.take(110)
    }

    private fun buildWhiteIntegral(table: Mat): WhiteIntegral {
        val width = table.cols()
        val height = table.rows()
        val channels = table.channels().coerceAtLeast(3)
        val bytes = ByteArray(width * height * channels)
        table.get(0, 0, bytes)

        val stride = width + 1
        val integral = IntArray((width + 1) * (height + 1))
        var y = 0
        while (y < height) {
            var rowSum = 0
            var x = 0
            while (x < width) {
                val i = (y * width + x) * channels
                val r = bytes[i].toInt() and 0xFF
                val g = bytes[i + 1].toInt() and 0xFF
                val b = bytes[i + 2].toInt() and 0xFF
                val hi = max(r, max(g, b))
                val lo = min(r, min(g, b))
                val white = if (hi >= 150 && hi - lo <= 58) 1 else 0
                rowSum += white
                integral[(y + 1) * stride + (x + 1)] = integral[y * stride + (x + 1)] + rowSum
                x++
            }
            y++
        }
        return WhiteIntegral(width, height, stride, integral)
    }

    private fun rectWhiteFraction(w: WhiteIntegral, cx: Int, cy: Int, radius: Int): Double {
        val x1 = (cx - radius).coerceIn(0, w.width)
        val y1 = (cy - radius).coerceIn(0, w.height)
        val x2 = (cx + radius + 1).coerceIn(0, w.width)
        val y2 = (cy + radius + 1).coerceIn(0, w.height)
        if (x2 <= x1 || y2 <= y1) return 0.0
        val sum = w.data[y2 * w.stride + x2] -
            w.data[y1 * w.stride + x2] -
            w.data[y2 * w.stride + x1] +
            w.data[y1 * w.stride + x1]
        return sum.toDouble() / ((x2 - x1) * (y2 - y1)).toDouble()
    }

    /** Large filled white patch = likely the cue ball endpoint, not the contact ring. */
    private fun maxSolidWhite(w: WhiteIntegral, p: Vec2, tableWidth: Double): Double {
        val search = max(10, (tableWidth * 0.025).toInt())
        val radius = max(4, (tableWidth * 0.008).toInt())
        val cx = p.x.toInt()
        val cy = p.y.toInt()
        var best = 0.0
        var yy = cy - search
        while (yy <= cy + search) {
            var xx = cx - search
            while (xx <= cx + search) {
                val f = rectWhiteFraction(w, xx, yy, radius)
                if (f > best) best = f
                xx += 2
            }
            yy += 2
        }
        return best
    }

    /** The real contact marker has a substantial amount of white guide/ring nearby. */
    private fun localWhiteDensity(w: WhiteIntegral, p: Vec2, tableWidth: Double): Double {
        val radius = max(10, (tableWidth * 0.022).toInt())
        return rectWhiteFraction(w, p.x.toInt(), p.y.toInt(), radius)
    }

    private fun findFork(
        segments: List<GuideSeg>,
        tableWidth: Double,
        whiteIntegral: WhiteIntegral
    ): Fork? {
        if (segments.isEmpty()) return null

        val longMin = max(52.0, tableWidth * 0.065)
        val branchMin = max(16.0, tableWidth * 0.016)
        val connectRadius = max(18.0, tableWidth * 0.024)
        val lineRadius = max(10.0, tableWidth * 0.015)

        var best: Fork? = null
        var bestScore = Double.NEGATIVE_INFINITY

        for (longSeg in segments.filter { it.length >= longMin }.take(30)) {
            val endpoints = listOf(longSeg.p1 to longSeg.p2, longSeg.p2 to longSeg.p1)
            for ((junction, otherEnd) in endpoints) {
                val localWhite = localWhiteDensity(whiteIntegral, junction, tableWidth)
                // Random anti-aliased line endpoints have almost no ring/guide mass nearby.
                if (localWhite < 0.075) continue

                val solidWhite = maxSolidWhite(whiteIntegral, junction, tableWidth)
                val incoming = (junction - otherEnd).normalized()
                if (incoming.length() < 0.5) continue

                val raw = mutableListOf<Branch>()
                for (seg in segments) {
                    if (seg === longSeg || seg.length < branchMin) continue
                    val d1 = (seg.p1 - junction).length()
                    val d2 = (seg.p2 - junction).length()
                    val nearest = min(d1, d2)
                    if (nearest > connectRadius) continue

                    val lineDist = distancePointToLine(junction, seg.p1, seg.p2)
                    if (lineDist > lineRadius) continue

                    val nearPoint: Vec2
                    val farPoint: Vec2
                    if (d1 <= d2) {
                        nearPoint = seg.p1
                        farPoint = seg.p2
                    } else {
                        nearPoint = seg.p2
                        farPoint = seg.p1
                    }

                    val direction = (farPoint - nearPoint).normalized()
                    if (direction.length() < 0.5) continue
                    val forward = direction.dot(incoming)
                    if (forward < -0.25) continue
                    if ((farPoint - junction).length() < branchMin * 0.80) continue

                    val quality = seg.length - nearest * 0.30 - lineDist * 1.15 +
                        if (forward > 0.92) tableWidth * 0.012 else 0.0
                    raw += Branch(direction, seg.length, quality)
                }

                val branches = deduplicateBranches(raw)
                if (branches.isEmpty()) continue
                val strongest = branches.first().quality
                val accepted = branches.filterIndexed { index, b ->
                    index == 0 || b.quality >= max(branchMin * 0.72, strongest * 0.40)
                }.take(2)
                if (accepted.isEmpty()) continue

                // This is the key cue-vs-contact discriminator. On the supplied frames the
                // cue endpoint has a ~0.8 solid-white score while the contact ring is ~0.45.
                val score = longSeg.length * 1.60 +
                    accepted.sumOf { it.quality } +
                    if (accepted.size >= 2) tableWidth * 0.05 else 0.0 -
                    solidWhite * tableWidth * 0.55 +
                    localWhite * tableWidth * 0.55

                if (score > bestScore) {
                    bestScore = score
                    best = Fork(junction, incoming, longSeg, accepted, solidWhite, localWhite)
                }
            }
        }
        return best
    }

    private fun deduplicateBranches(raw: List<Branch>): List<Branch> {
        val kept = mutableListOf<Branch>()
        for (candidate in raw.sortedByDescending { it.quality }) {
            if (kept.none { branchAngle(it.direction, candidate.direction) < 8.0 }) kept += candidate
        }
        return kept
    }

    private fun distancePointToLine(p: Vec2, a: Vec2, b: Vec2): Double {
        val ab = b - a
        val len = ab.length()
        if (len < 1e-9) return (p - a).length()
        return abs(ab.x * (a.y - p.y) - (a.x - p.x) * ab.y) / len
    }

    private fun extendBranch(
        startLocal: Vec2,
        direction0: Vec2,
        tableRect: Rect,
        scale: Double,
        firstKind: SegmentKind,
        bounces: Int
    ): List<TrajectorySegment> {
        val result = mutableListOf<TrajectorySegment>()
        val radius = tableRect.width * 0.012
        val verticalInset = tableRect.height * 0.018
        val bounds = PlayRect(
            radius,
            radius + verticalInset,
            tableRect.width - radius,
            tableRect.height - radius - verticalInset
        )

        var p = startLocal
        var d = direction0.normalized()
        repeat(max(1, bounces + 1)) { index ->
            val hit = rayToBoundary(p, d, bounds) ?: return@repeat
            result += TrajectorySegment(
                toGlobal(p, tableRect, scale),
                toGlobal(hit.first, tableRect, scale),
                if (index == 0) firstKind else SegmentKind.BOUNCE,
                firstKind == SegmentKind.CUE_AFTER || index > 0
            )
            if (index < bounces) {
                d = Vec2(if (hit.second) -d.x else d.x, if (hit.third) -d.y else d.y)
                p = hit.first + d * 0.8
            }
        }
        return result
    }

    private fun rayToBoundary(
        start: Vec2,
        d0: Vec2,
        rect: PlayRect
    ): Triple<Vec2, Boolean, Boolean>? {
        val d = d0.normalized()
        val eps = 1e-7
        val ts = mutableListOf<Double>()
        if (abs(d.x) > eps) {
            ((rect.left - start.x) / d.x).takeIf { it > eps }?.let(ts::add)
            ((rect.right - start.x) / d.x).takeIf { it > eps }?.let(ts::add)
        }
        if (abs(d.y) > eps) {
            ((rect.top - start.y) / d.y).takeIf { it > eps }?.let(ts::add)
            ((rect.bottom - start.y) / d.y).takeIf { it > eps }?.let(ts::add)
        }
        if (ts.isEmpty()) return null
        val t = ts.min()
        val q = start + d * t
        val p = Vec2(q.x.coerceIn(rect.left, rect.right), q.y.coerceIn(rect.top, rect.bottom))
        val hx = abs(p.x - rect.left) < 2.0 || abs(p.x - rect.right) < 2.0
        val hy = abs(p.y - rect.top) < 2.0 || abs(p.y - rect.bottom) < 2.0
        return Triple(p, hx, hy)
    }

    private fun toGlobal(local: Vec2, tableRect: Rect, scale: Double): Vec2 {
        return Vec2((local.x + tableRect.x) / scale, (local.y + tableRect.y) / scale)
    }

    private fun branchAngle(a0: Vec2, b0: Vec2): Double {
        val a = a0.normalized()
        val b = b0.normalized()
        return Math.toDegrees(acos(a.dot(b).coerceIn(-1.0, 1.0)))
    }
}
