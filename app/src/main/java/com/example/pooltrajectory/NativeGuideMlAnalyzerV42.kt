package com.example.pooltrajectory

import android.content.Context
import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * 4.2 geometry post-pass.
 *
 * 4.1/ML is kept as the detector of the contact scene and short native branches.
 * 4.2 then re-fits the visible short white native segment directly on the original
 * captured pixels and extends that exact axis to the cushion. When two branches are
 * present, their object/cue identity is rechecked frame-by-frame from a local target
 * ball circle near the intersection; there is no cross-frame identity fallback.
 */
class NativeGuideMlAnalyzerV42(private val context: Context) {
    private val base = NativeGuideMlAnalyzerV41(context)
    private val prefs = context.getSharedPreferences(MainActivity.PREFS, Context.MODE_PRIVATE)

    private data class Fit(val dir: Vec2, val center: Vec2, val rms: Double, val linearity: Double)
    private data class RefinedBranch(
        val originalKind: SegmentKind,
        val start: Vec2,
        val dir: Vec2,
        val quality: Double
    )
    private data class Assignment(
        val objectBranch: RefinedBranch?,
        val cueBranch: RefinedBranch?,
        val junction: Vec2?,
        val targetCenter: Vec2?,
        val quality: Double,
        val retyped: Boolean
    )

    fun analyze(bitmap: Bitmap): AnalysisResult {
        val old = base.analyze(bitmap)
        val play = old.playRect ?: return old.copy(status = old.status.replace("ML 4.1", "ML 4.2"))
        val raw = old.segments.filter { it.kind == SegmentKind.OBJECT || it.kind == SegmentKind.CUE_AFTER }.take(2)
        if (raw.isEmpty()) return old.copy(status = old.status.replace("ML 4.1", "ML 4.2"))

        val mask = buildWhiteMask(bitmap)
        val refined = raw.mapNotNull { refineNativeShortBranch(mask, it, play) }
        mask.release()
        if (refined.isEmpty()) {
            return old.copy(
                segments = old.segments.filter { it.kind == SegmentKind.AIM },
                confidence = min(old.confidence, 45),
                status = "ML 4.2: короткая ветка не подтверждена"
            )
        }

        val assignment = classify(bitmap, refined, play)
        val out = mutableListOf<TrajectorySegment>()
        if (prefs.getBoolean(MainActivity.KEY_DEBUG, false)) {
            out += old.segments.filter { it.kind == SegmentKind.AIM }
            refined.forEach { b ->
                val length = min(42.0, playWidth(play) * 0.040)
                out += TrajectorySegment(b.start - b.dir * length, b.start, SegmentKind.AIM)
            }
            assignment.junction?.let { j ->
                val r = max(7.0, playWidth(play) * 0.007)
                out += TrajectorySegment(Vec2(j.x-r,j.y), Vec2(j.x+r,j.y), SegmentKind.AIM)
                out += TrajectorySegment(Vec2(j.x,j.y-r), Vec2(j.x,j.y+r), SegmentKind.AIM)
            }
            assignment.targetCenter?.let { c ->
                val r = max(6.0, playWidth(play) * 0.006)
                out += TrajectorySegment(Vec2(c.x-r,c.y), Vec2(c.x+r,c.y), SegmentKind.AIM)
                out += TrajectorySegment(Vec2(c.x,c.y-r), Vec2(c.x,c.y+r), SegmentKind.AIM)
            }
        }

        assignment.objectBranch?.let { b ->
            val bounces = prefs.getInt(MainActivity.KEY_BOUNCES, 0).coerceIn(0, 2)
            out += extendToTable(b.start, b.dir, play, SegmentKind.OBJECT, bounces)
        }
        assignment.cueBranch?.let { b ->
            out += extendToTable(b.start, b.dir, play, SegmentKind.CUE_AFTER, 0)
        }

        val status = when {
            assignment.objectBranch != null && assignment.cueBranch != null && assignment.retyped -> "ML 4.2: объект + биток • перепроверено"
            assignment.objectBranch != null && assignment.cueBranch != null -> "ML 4.2: объект + биток"
            assignment.objectBranch != null -> "ML 4.2: прицельный шар"
            assignment.cueBranch != null -> "ML 4.2: белый шар"
            else -> "ML 4.2: тип ветки не подтверждён"
        }
        val confidence = ((refined.map { it.quality }.average() * 0.62 + assignment.quality * 0.38) * 100.0)
            .toInt().coerceIn(45, 99)
        return old.copy(segments = out, confidence = confidence, status = status)
    }

    /**
     * The old long coloured trajectory begins exactly at the outer end of the short
     * native branch. Trace backwards from that point only through achromatic white
     * pixels, stop at the first real gap, then fit the collected high-resolution pixels.
     */
    private fun refineNativeShortBranch(mask: Mat, raw: TrajectorySegment, play: PlayRect): RefinedBranch? {
        val dir0 = (raw.end - raw.start).normalized()
        if (dir0.length() < 0.5) return null
        val perp = Vec2(-dir0.y, dir0.x)
        val width = playWidth(play)
        val corridor = max(3.0, width * 0.0045)
        val maxBack = min(92.0, max(52.0, width * 0.070))
        val step = 1.25

        var t = 0.0
        var started = false
        var firstT = 0.0
        var lastT = 0.0
        var initialMiss = 0
        var missRun = 0
        while (t <= maxBack) {
            val p = raw.start - dir0 * t
            val hit = whiteAcross(mask, p, perp, corridor)
            if (hit) {
                if (!started) { started = true; firstT = t }
                lastT = t
                missRun = 0
            } else if (!started) {
                initialMiss++
                if (initialMiss > 9) return null
            } else {
                missRun++
                if (missRun > 6) break
            }
            t += step
        }
        if (!started || lastT - firstT < max(8.0, width * 0.009)) return null

        val points = mutableListOf<Vec2>()
        var tt = max(0.0, firstT - 2.0)
        while (tt <= min(maxBack, lastT + 2.0)) {
            val center = raw.start - dir0 * tt
            var o = -corridor
            while (o <= corridor) {
                val q = center + perp * o
                val x = q.x.toInt(); val y = q.y.toInt()
                if (x in 0 until mask.cols() && y in 0 until mask.rows()) {
                    val v = mask.get(y, x)
                    if (v != null && v.isNotEmpty() && v[0] > 0.0) points += Vec2(x.toDouble(), y.toDouble())
                }
                o += 1.0
            }
            tt += 1.0
        }
        if (points.size < 18) return null

        val fit = fitPoints(points, dir0) ?: return null
        val delta = branchAngle(fit.dir, dir0)
        if (delta > 7.0 || fit.linearity < 3.0 || fit.rms > max(3.8, width * 0.0048)) return null

        val projections = points.map { (it - raw.start).dot(fit.dir) }.sorted()
        val outer = quantile(projections, 0.97)
        val inner = quantile(projections, 0.08)
        val visibleLength = outer - inner
        if (visibleLength < max(8.0, width * 0.009)) return null
        val start = raw.start + fit.dir * outer
        val linearQ = ((fit.linearity - 2.0) / 10.0).coerceIn(0.25, 1.0)
        val residualQ = exp(-fit.rms * 0.30).coerceIn(0.20, 1.0)
        val angleQ = exp(-delta * 0.20).coerceIn(0.25, 1.0)
        val quality = (linearQ * 0.42 + residualQ * 0.36 + angleQ * 0.22).coerceIn(0.0, 1.0)
        return RefinedBranch(raw.kind, start, fit.dir, quality)
    }

    private fun whiteAcross(mask: Mat, p: Vec2, perp: Vec2, corridor: Double): Boolean {
        var hits = 0
        var samples = 0
        var o = -corridor
        while (o <= corridor) {
            val q = p + perp * o
            val x = q.x.toInt(); val y = q.y.toInt()
            if (x in 0 until mask.cols() && y in 0 until mask.rows()) {
                samples++
                val v = mask.get(y, x)
                if (v != null && v.isNotEmpty() && v[0] > 0.0) hits++
            }
            o += 1.5
        }
        return hits >= max(2, samples / 4)
    }

    private fun classify(bitmap: Bitmap, branches: List<RefinedBranch>, play: PlayRect): Assignment {
        if (branches.size == 1) {
            val b = branches.first()
            return if (b.originalKind == SegmentKind.OBJECT) {
                Assignment(b, null, null, null, b.quality, false)
            } else {
                Assignment(null, b, null, null, b.quality, false)
            }
        }

        val b0 = branches[0]
        val b1 = branches[1]
        val junction = lineIntersection(b0.start, b0.dir, b1.start, b1.dir)
        if (junction == null ||
            (junction - b0.start).length() > playWidth(play) * 0.12 ||
            (junction - b1.start).length() > playWidth(play) * 0.12) {
            return preserveOldKinds(branches, null, null)
        }

        val evidence = objectEvidence(bitmap, junction, branches, play)
        if (evidence == null) return preserveOldKinds(branches, junction, null)
        val (objectIndex, targetCenter, score, margin) = evidence
        if (score < 0.58 || margin < 0.10) return preserveOldKinds(branches, junction, targetCenter)

        val obj = branches[objectIndex]
        val cue = branches[1 - objectIndex]
        val quality = (obj.quality * 0.34 + cue.quality * 0.26 + score * 0.40).coerceIn(0.0, 1.0)
        return Assignment(obj, cue, junction, targetCenter, quality, true)
    }

    private fun preserveOldKinds(branches: List<RefinedBranch>, junction: Vec2?, target: Vec2?): Assignment {
        val obj = branches.firstOrNull { it.originalKind == SegmentKind.OBJECT }
        val cue = branches.firstOrNull { it.originalKind == SegmentKind.CUE_AFTER }
        val q = branches.map { it.quality }.average() * 0.78
        return Assignment(obj, cue, junction, target, q.coerceIn(0.0, 1.0), false)
    }

    /**
     * Search circles only near the two fitted branch axes. The contact ring itself is
     * rejected by distance from the branch intersection. A strong score on one branch
     * is required before 4.2 is allowed to swap yellow/blue identity from 4.1.
     */
    private fun objectEvidence(
        bitmap: Bitmap,
        junction: Vec2,
        branches: List<RefinedBranch>,
        play: PlayRect
    ): Quadruple? {
        val width = playWidth(play)
        val search = min(115.0, max(72.0, width * 0.090)).toInt()
        val x1 = max(play.left.toInt(), junction.x.toInt() - search)
        val y1 = max(play.top.toInt(), junction.y.toInt() - search)
        val x2 = min(play.right.toInt(), junction.x.toInt() + search + 1)
        val y2 = min(play.bottom.toInt(), junction.y.toInt() + search + 1)
        if (x2 - x1 < 45 || y2 - y1 < 45) return null

        val src = Mat(); Utils.bitmapToMat(bitmap, src)
        val crop = src.submat(Rect(x1, y1, x2 - x1, y2 - y1))
        val gray = Mat(); Imgproc.cvtColor(crop, gray, Imgproc.COLOR_RGBA2GRAY)
        crop.release(); src.release()
        Imgproc.medianBlur(gray, gray, 5)
        val circles = Mat()
        val minR = max(7, (width * 0.009).toInt())
        val maxR = max(minR + 3, (width * 0.023).toInt())
        Imgproc.HoughCircles(
            gray, circles, Imgproc.HOUGH_GRADIENT, 1.2,
            max(16.0, width * 0.020), 105.0, 14.0, minR, maxR
        )
        gray.release()

        val best = DoubleArray(2) { Double.NEGATIVE_INFINITY }
        val centers = arrayOfNulls<Vec2>(2)
        val count = if (circles.rows() > 0) circles.cols() else 0
        for (i in 0 until count) {
            val c = circles.get(0, i) ?: continue
            if (c.size < 3) continue
            val center = Vec2(x1 + c[0], y1 + c[1])
            val radius = c[2]
            val v = center - junction
            val d = v.length()
            if (d < radius * 1.35 || d > radius * 3.15) continue
            val targetDir = v.normalized()
            for (bi in 0..1) {
                val align = branches[bi].dir.dot(targetDir)
                if (align < 0.82) continue
                val diameterError = abs(d / max(1.0, radius * 2.0) - 1.0)
                val distanceQ = exp(-diameterError * 2.1)
                val alignQ = ((align - 0.82) / 0.18).coerceIn(0.0, 1.0)
                val startDistance = (center - branches[bi].start).length()
                val proximityQ = exp(-startDistance / max(18.0, radius * 2.1))
                val q = (alignQ * 0.50 + distanceQ * 0.34 + proximityQ * 0.16).coerceIn(0.0, 1.0)
                if (q > best[bi]) { best[bi] = q; centers[bi] = center }
            }
        }
        circles.release()

        val index = if (best[0] >= best[1]) 0 else 1
        if (!best[index].isFinite()) return null
        val other = if (best[1-index].isFinite()) best[1-index] else 0.0
        return Quadruple(index, centers[index] ?: return null, best[index], best[index] - other)
    }

    private data class Quadruple(val index: Int, val center: Vec2, val score: Double, val margin: Double)

    private fun extendToTable(
        start: Vec2,
        direction0: Vec2,
        play: PlayRect,
        firstKind: SegmentKind,
        bounces: Int
    ): List<TrajectorySegment> {
        val out = mutableListOf<TrajectorySegment>()
        val w = playWidth(play); val h = play.bottom - play.top
        val bounds = PlayRect(
            play.left + w * 0.0145,
            play.top + h * 0.021,
            play.right - w * 0.0145,
            play.bottom - h * 0.021
        )
        var p = start
        var d = direction0.normalized()
        if (d.length() < 0.5) return out
        for (i in 0..bounces) {
            val hit = rayToBounds(p, d, bounds) ?: break
            out += TrajectorySegment(p, hit, if (i == 0) firstKind else SegmentKind.BOUNCE)
            if (i == bounces) break
            var nx = d.x; var ny = d.y
            val eps = 3.0
            if (abs(hit.x-bounds.left) < eps || abs(hit.x-bounds.right) < eps) nx = -nx
            if (abs(hit.y-bounds.top) < eps || abs(hit.y-bounds.bottom) < eps) ny = -ny
            d = Vec2(nx, ny).normalized()
            p = hit + d * 1.5
        }
        return out
    }

    private fun buildWhiteMask(bitmap: Bitmap): Mat {
        val rgba = Mat(); Utils.bitmapToMat(bitmap, rgba)
        val rgb = Mat(); val hsv = Mat(); val mask = Mat()
        Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB); rgba.release()
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV); rgb.release()
        Core.inRange(hsv, Scalar(0.0, 0.0, 152.0), Scalar(179.0, 84.0, 255.0), mask)
        hsv.release()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0,2.0))
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel); kernel.release()
        return mask
    }

    private fun fitPoints(points: List<Vec2>, desired: Vec2): Fit? {
        if (points.size < 3) return null
        val mx = points.sumOf { it.x } / points.size
        val my = points.sumOf { it.y } / points.size
        var xx=0.0; var xy=0.0; var yy=0.0
        for (p in points) {
            val dx=p.x-mx; val dy=p.y-my
            xx += dx*dx; xy += dx*dy; yy += dy*dy
        }
        val trace=xx+yy
        val disc=sqrt(max(0.0,(xx-yy)*(xx-yy)+4.0*xy*xy))
        val l1=(trace+disc)*0.5
        val l2=max(1e-6,(trace-disc)*0.5)
        if (l1 <= 1e-6) return null
        val angle=0.5*atan2(2.0*xy,xx-yy)
        var dir=Vec2(cos(angle),sin(angle)).normalized()
        if (dir.dot(desired) < 0.0) dir = dir * -1.0
        var residual=0.0
        for (p in points) {
            val v=p-Vec2(mx,my)
            val d=abs(v.x*dir.y-v.y*dir.x)
            residual += d*d
        }
        return Fit(dir,Vec2(mx,my),sqrt(residual/points.size),l1/l2)
    }

    private fun lineIntersection(p1: Vec2, d1: Vec2, p2: Vec2, d2: Vec2): Vec2? {
        val den = d1.x*d2.y - d1.y*d2.x
        if (abs(den) < 0.07) return null
        val delta = p2-p1
        val t = (delta.x*d2.y - delta.y*d2.x) / den
        val hit = p1 + d1*t
        return if (hit.x.isFinite() && hit.y.isFinite()) hit else null
    }

    private fun rayToBounds(p:Vec2,d:Vec2,b:PlayRect):Vec2? {
        var best=Double.POSITIVE_INFINITY
        fun test(t:Double) {
            if (t<=1e-6 || t>=best) return
            val x=p.x+d.x*t; val y=p.y+d.y*t
            if (x>=b.left-0.5 && x<=b.right+0.5 && y>=b.top-0.5 && y<=b.bottom+0.5) best=t
        }
        if (abs(d.x)>1e-9) { test((b.left-p.x)/d.x); test((b.right-p.x)/d.x) }
        if (abs(d.y)>1e-9) { test((b.top-p.y)/d.y); test((b.bottom-p.y)/d.y) }
        return if (best.isFinite()) p+d*best else null
    }

    private fun quantile(sorted: List<Double>, q: Double): Double {
        if (sorted.isEmpty()) return 0.0
        return sorted[(sorted.lastIndex*q).toInt().coerceIn(0,sorted.lastIndex)]
    }

    private fun branchAngle(a:Vec2,b:Vec2):Double {
        val na=a.normalized(); val nb=b.normalized()
        return Math.toDegrees(acos(na.dot(nb).coerceIn(-1.0,1.0)))
    }

    private fun playWidth(p: PlayRect) = p.right - p.left
}
