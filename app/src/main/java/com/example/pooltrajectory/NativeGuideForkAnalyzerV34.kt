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
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * v3.4 keeps Hough only for the long incoming native guide.
 * Short post-contact branches are read locally from white pixels leaving the native ring.
 */
class NativeGuideForkAnalyzerV34(private val context: Context) {
    private val prefs = context.getSharedPreferences(MainActivity.PREFS, Context.MODE_PRIVATE)

    private data class GuideSeg(val p1: Vec2, val p2: Vec2, val length: Double)
    private data class RayBranch(
        val direction: Vec2,
        val start: Vec2,
        val end: Vec2,
        val score: Double,
        val length: Double
    )
    private data class Fork(
        val junction: Vec2,
        val incoming: Vec2,
        val incomingSeg: GuideSeg,
        val branches: List<RayBranch>
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
            return AnalysisResult(bitmap.width, bitmap.height, null, emptyList(), null, emptyList(), null, 0, "стол не подтвержден")
        }

        val table = work.submat(tableRect)
        val mask = buildWhiteMask(table)
        val longSegments = detectLongGuideSegments(mask, table.cols())
        val fork = findFork(mask, longSegments, table.cols().toDouble())

        val playRect = PlayRect(
            tableRect.x / scale,
            tableRect.y / scale,
            (tableRect.x + tableRect.width) / scale,
            (tableRect.y + tableRect.height) / scale
        )

        val out = mutableListOf<TrajectorySegment>()
        val debug = prefs.getBoolean(MainActivity.KEY_DEBUG, false)

        if (fork != null) {
            val ordered = fork.branches.sortedBy { branchAngle(fork.incoming, it.direction) }.take(2)
            ordered.forEachIndexed { index, branch ->
                val kind = if (index == 0) SegmentKind.OBJECT else SegmentKind.CUE_AFTER
                val bounces = if (kind == SegmentKind.OBJECT) prefs.getInt(MainActivity.KEY_BOUNCES, 0).coerceIn(0, 2) else 0
                out += extendBranch(branch.end, branch.direction, tableRect, scale, kind, bounces)

                if (debug) {
                    out += TrajectorySegment(
                        toGlobal(branch.start, tableRect, scale),
                        toGlobal(branch.end, tableRect, scale),
                        SegmentKind.AIM
                    )
                }
            }

            if (debug) {
                out += TrajectorySegment(
                    toGlobal(fork.incomingSeg.p1, tableRect, scale),
                    toGlobal(fork.incomingSeg.p2, tableRect, scale),
                    SegmentKind.AIM
                )
                val j = fork.junction
                val cross = max(7.0, tableRect.width * 0.009)
                out += TrajectorySegment(toGlobal(Vec2(j.x-cross,j.y),tableRect,scale),toGlobal(Vec2(j.x+cross,j.y),tableRect,scale),SegmentKind.AIM)
                out += TrajectorySegment(toGlobal(Vec2(j.x,j.y-cross),tableRect,scale),toGlobal(Vec2(j.x,j.y+cross),tableRect,scale),SegmentKind.AIM)
            }
        } else if (debug) {
            longSegments.take(8).forEach { s ->
                out += TrajectorySegment(toGlobal(s.p1,tableRect,scale),toGlobal(s.p2,tableRect,scale),SegmentKind.AIM)
            }
        }

        mask.release()
        table.release()
        work.release()

        val status = when {
            longSegments.isEmpty() -> "длинная штатная линия не найдена"
            fork == null -> "короткая ветка у кружка не найдена"
            fork.branches.size == 1 -> "локальная ветка найдена • 1"
            else -> "локальные ветки найдены • 2"
        }
        val confidence = when {
            longSegments.isEmpty() -> 25
            fork == null -> 45
            fork.branches.size == 1 -> 90
            else -> 98
        }
        return AnalysisResult(bitmap.width, bitmap.height, null, emptyList(), null, out, playRect, confidence, status)
    }

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
        if (valid < 180) { hsv.release(); return null }
        val dominant = hist.indices.maxByOrNull { hist[it] } ?: run { hsv.release(); return null }
        var cluster = 0
        for (d in -11..11) {
            var h = dominant + d
            if (h < 0) h += 180
            if (h >= 180) h -= 180
            cluster += hist[h]
        }
        hsv.release()
        val share = cluster.toDouble() / valid.toDouble()
        return if (dominant in 35..125 && share >= 0.42) rect else null
    }

    private fun buildWhiteMask(table: Mat): Mat {
        val rgb = Mat()
        val hsv = Mat()
        val mask = Mat()
        Imgproc.cvtColor(table, rgb, Imgproc.COLOR_RGBA2RGB)
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
        rgb.release()
        Core.inRange(hsv, Scalar(0.0, 0.0, 150.0), Scalar(179.0, 82.0, 255.0), mask)
        hsv.release()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0, 2.0))
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel)
        kernel.release()
        val border = max(5, (table.cols() * 0.008).toInt())
        Imgproc.rectangle(mask, Point(0.0,0.0), Point((mask.cols()-1).toDouble(),(mask.rows()-1).toDouble()), Scalar(0.0), border)
        return mask
    }

    private fun detectLongGuideSegments(mask: Mat, tableWidth: Int): List<GuideSeg> {
        val lines = Mat()
        val minLen = max(50.0, tableWidth * 0.065)
        Imgproc.HoughLinesP(mask, lines, 1.0, Math.PI / 360.0, 24, minLen, 18.0)
        val out = mutableListOf<GuideSeg>()
        for (row in 0 until lines.rows()) {
            val l = lines.get(row,0) ?: continue
            if (l.size < 4) continue
            val p1 = Vec2(l[0],l[1])
            val p2 = Vec2(l[2],l[3])
            val len = (p2-p1).length()
            if (len >= minLen) out += GuideSeg(p1,p2,len)
        }
        lines.release()
        return out.sortedByDescending { it.length }.take(18)
    }

    private fun findFork(mask: Mat, segments: List<GuideSeg>, tableWidth: Double): Fork? {
        if (segments.isEmpty()) return null
        var best: Fork? = null
        var bestScore = Double.NEGATIVE_INFINITY

        for (seg in segments) {
            val endpoints = listOf(seg.p1 to seg.p2, seg.p2 to seg.p1)
            for ((junction, otherEnd) in endpoints) {
                if (!inside(mask,junction,4)) continue
                val incoming = (junction-otherEnd).normalized()
                if (incoming.length() < 0.5) continue
                val branches = scanBranches(mask, junction, incoming, tableWidth)
                if (branches.isEmpty()) continue
                val ring = whiteFraction(mask,junction,max(8,(tableWidth*0.014).toInt()))
                val score = seg.length * 0.55 + branches.sumOf { it.score } + ring * tableWidth * 0.08 + if (branches.size >= 2) tableWidth * 0.04 else 0.0
                if (score > bestScore) {
                    bestScore = score
                    best = Fork(junction,incoming,seg,branches)
                }
            }
        }

        val seed = best ?: return null
        var refined = seed
        var refinedScore = seed.branches.sumOf { it.score }
        val offsets = intArrayOf(-6,-3,0,3,6)
        for (dy in offsets) for (dx in offsets) {
            val j = Vec2(seed.junction.x+dx, seed.junction.y+dy)
            if (!inside(mask,j,4)) continue
            val other = if ((seed.incomingSeg.p1-seed.junction).length() > (seed.incomingSeg.p2-seed.junction).length()) seed.incomingSeg.p1 else seed.incomingSeg.p2
            val incoming = (j-other).normalized()
            val branches = scanBranches(mask,j,incoming,tableWidth)
            if (branches.isEmpty()) continue
            val score = branches.sumOf { it.score } - (j-seed.junction).length()*0.4 + if (branches.size >= 2) tableWidth*0.02 else 0.0
            if (score > refinedScore) {
                refinedScore = score
                refined = Fork(j,incoming,seed.incomingSeg,branches)
            }
        }
        return refined
    }

    private fun scanBranches(mask: Mat, junction: Vec2, incoming: Vec2, tableWidth: Double): List<RayBranch> {
        val inner = max(13.0, tableWidth * 0.019)
        val outer = max(46.0, tableWidth * 0.080)
        val minLength = max(12.0, tableWidth * 0.016)
        val raw = mutableListOf<RayBranch>()

        var deg = 0
        while (deg < 360) {
            val a = Math.toRadians(deg.toDouble())
            val dir = Vec2(cos(a),sin(a))
            // Native post-contact branches go forward or sideways. The ray back to the cue ball is excluded.
            if (dir.dot(incoming) >= -0.08) {
                traceRay(mask,junction,dir,inner,outer)?.let { branch ->
                    if (branch.length >= minLength) raw += branch
                }
            }
            deg += 2
        }

        val kept = mutableListOf<RayBranch>()
        for (c in raw.sortedByDescending { it.score }) {
            if (kept.none { branchAngle(it.direction,c.direction) < 11.0 }) kept += c
            if (kept.size >= 4) break
        }
        if (kept.isEmpty()) return emptyList()
        val strongest = kept.first().score
        return kept.filterIndexed { index,b -> index == 0 || b.score >= strongest * 0.55 }.take(2)
    }

    private fun traceRay(mask: Mat, junction: Vec2, dir: Vec2, inner: Double, outer: Double): RayBranch? {
        val perp = Vec2(-dir.y,dir.x)
        val step = 1.5
        var r = inner
        var started = false
        var first = 0.0
        var last = 0.0
        var hits = 0
        var samples = 0
        var missRun = 0
        var initialMiss = 0

        while (r <= outer) {
            val p = junction + dir * r
            val white = whiteAt(mask,p,perp)
            samples++
            if (white) {
                if (!started) {
                    started = true
                    first = r
                }
                last = r
                hits++
                missRun = 0
            } else if (!started) {
                initialMiss++
                if (initialMiss > 5) return null
            } else {
                missRun++
                if (missRun > 3) break
            }
            r += step
        }
        if (!started || last <= first) return null
        val length = last-first
        val hitRatio = hits.toDouble() / max(1,samples).toDouble()
        val score = length + hitRatio*22.0 - initialMiss*1.2
        return RayBranch(dir,junction+dir*first,junction+dir*last,score,length)
    }

    private fun whiteAt(mask: Mat, p: Vec2, perp: Vec2): Boolean {
        val offsets = doubleArrayOf(-2.0,-1.0,0.0,1.0,2.0)
        var hits = 0
        for (o in offsets) {
            val q = p + perp*o
            val x = q.x.toInt()
            val y = q.y.toInt()
            if (x !in 0 until mask.cols() || y !in 0 until mask.rows()) continue
            val v = mask.get(y,x)
            if (v != null && v.isNotEmpty() && v[0] > 0.0) hits++
        }
        return hits >= 2
    }

    private fun whiteFraction(mask: Mat, p: Vec2, radius: Int): Double {
        val x1 = max(0,p.x.toInt()-radius)
        val y1 = max(0,p.y.toInt()-radius)
        val x2 = min(mask.cols(),p.x.toInt()+radius+1)
        val y2 = min(mask.rows(),p.y.toInt()+radius+1)
        if (x2 <= x1 || y2 <= y1) return 0.0
        val roi = mask.submat(Rect(x1,y1,x2-x1,y2-y1))
        val count = Core.countNonZero(roi)
        roi.release()
        return count.toDouble()/((x2-x1)*(y2-y1)).toDouble()
    }

    private fun inside(mask: Mat, p: Vec2, margin: Int): Boolean =
        p.x >= margin && p.y >= margin && p.x < mask.cols()-margin && p.y < mask.rows()-margin

    private fun extendBranch(startLocal: Vec2, direction0: Vec2, tableRect: Rect, scale: Double, firstKind: SegmentKind, bounces: Int): List<TrajectorySegment> {
        val result = mutableListOf<TrajectorySegment>()
        val radius = tableRect.width * 0.012
        val bounds = PlayRect(radius,radius,tableRect.width-radius,tableRect.height-radius)
        var p = startLocal
        var d = direction0.normalized()
        if (d.length() < 0.5) return result

        for (i in 0..bounces) {
            val hit = rayToBounds(p,d,bounds) ?: break
            val kind = if (i == 0) firstKind else SegmentKind.BOUNCE
            result += TrajectorySegment(toGlobal(p,tableRect,scale),toGlobal(hit,tableRect,scale),kind)
            if (i == bounces) break
            var nx = d.x
            var ny = d.y
            val eps = 2.5
            if (abs(hit.x-bounds.left) < eps || abs(hit.x-bounds.right) < eps) nx = -nx
            if (abs(hit.y-bounds.top) < eps || abs(hit.y-bounds.bottom) < eps) ny = -ny
            d = Vec2(nx,ny).normalized()
            p = hit + d*1.5
        }
        return result
    }

    private fun rayToBounds(p: Vec2, d: Vec2, b: PlayRect): Vec2? {
        var best = Double.POSITIVE_INFINITY
        fun test(t: Double) {
            if (t <= 1e-6 || t >= best) return
            val x = p.x+d.x*t
            val y = p.y+d.y*t
            if (x >= b.left-0.5 && x <= b.right+0.5 && y >= b.top-0.5 && y <= b.bottom+0.5) best = t
        }
        if (abs(d.x) > 1e-9) { test((b.left-p.x)/d.x); test((b.right-p.x)/d.x) }
        if (abs(d.y) > 1e-9) { test((b.top-p.y)/d.y); test((b.bottom-p.y)/d.y) }
        return if (best.isFinite()) p+d*best else null
    }

    private fun toGlobal(p: Vec2, tableRect: Rect, scale: Double): Vec2 =
        Vec2((tableRect.x+p.x)/scale,(tableRect.y+p.y)/scale)

    private fun branchAngle(a: Vec2, b: Vec2): Double {
        val na=a.normalized(); val nb=b.normalized()
        return Math.toDegrees(acos(na.dot(nb).coerceIn(-1.0,1.0)))
    }
}
