package com.example.pooltrajectory

import android.content.Context
import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
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
 * v3.1: trusts only a real felt contour and a physically connected native guide fork.
 * Straight-through object-ball branches are valid (important for break/head-on shots).
 */
class NativeGuideForkAnalyzerV31(private val context: Context) {
    private val prefs = context.getSharedPreferences(MainActivity.PREFS, Context.MODE_PRIVATE)

    private data class GuideSeg(val p1: Vec2, val p2: Vec2, val length: Double)
    private data class Branch(
        val direction: Vec2,
        val farPoint: Vec2,
        val length: Double,
        val quality: Double
    )
    private data class Fork(
        val junction: Vec2,
        val incoming: Vec2,
        val incomingSeg: GuideSeg,
        val branches: List<Branch>
    )

    fun analyze(bitmap: Bitmap): AnalysisResult {
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)
        val scale = min(1.0, 1280.0 / src.cols().toDouble())
        val work = Mat()
        if (scale < 0.999) Imgproc.resize(src, work, Size(), scale, scale, Imgproc.INTER_AREA)
        else src.copyTo(work)
        src.release()

        val tableRect = detectFeltRect(work)
        if (tableRect == null) {
            work.release()
            return AnalysisResult(
                bitmap.width, bitmap.height, null, emptyList(), null,
                emptyList(), null, 0, "стол не найден"
            )
        }

        val table = work.submat(tableRect)
        val segments = detectGuideSegments(table)
        val fork = findFork(segments, table.cols().toDouble())
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
                out += extendBranch(branch.farPoint, branch.direction, tableRect, scale, kind, bounces)
            }

            if (debug) {
                out += TrajectorySegment(
                    toGlobal(fork.incomingSeg.p1, tableRect, scale),
                    toGlobal(fork.incomingSeg.p2, tableRect, scale),
                    SegmentKind.AIM
                )
                val j = fork.junction
                val cross = max(7.0, tableRect.width * 0.008)
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
            segments.take(12).forEach { s ->
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
            fork == null -> "связная развилка не найдена"
            fork.branches.size == 1 -> "1 связная ветка"
            else -> "2 связные ветки"
        }
        val confidence = when {
            segments.isEmpty() -> 25
            fork == null -> 45
            fork.branches.size == 1 -> 80
            else -> 98
        }

        return AnalysisResult(
            bitmap.width, bitmap.height, null, emptyList(), null,
            out, playRect, confidence, status
        )
    }

    /** Returns null instead of inventing a table rectangle on loading/menu screens. */
    private fun detectFeltRect(work: Mat): Rect? {
        val rgb = Mat()
        val hsv = Mat()
        Imgproc.cvtColor(work, rgb, Imgproc.COLOR_RGBA2RGB)
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
        rgb.release()

        val x0 = (work.cols() * 0.18).toInt().coerceIn(0, work.cols() - 1)
        val x1 = (work.cols() * 0.82).toInt().coerceIn(x0 + 1, work.cols())
        val y0 = (work.rows() * 0.22).toInt().coerceIn(0, work.rows() - 1)
        val y1 = (work.rows() * 0.88).toInt().coerceIn(y0 + 1, work.rows())

        val hist = IntArray(180)
        var valid = 0
        var yy = y0
        while (yy < y1) {
            var xx = x0
            while (xx < x1) {
                val p = hsv.get(yy, xx)
                if (p != null && p.size >= 3 && p[1] >= 60.0 && p[2] >= 65.0) {
                    hist[p[0].toInt().coerceIn(0, 179)]++
                    valid++
                }
                xx += 4
            }
            yy += 4
        }
        if (valid < 250) {
            hsv.release()
            return null
        }

        val dominantHue = hist.indices.maxByOrNull { hist[it] } ?: run {
            hsv.release(); return null
        }
        if (hist[dominantHue] < max(80, valid / 25)) {
            hsv.release()
            return null
        }

        val mask = Mat()
        val low = dominantHue - 13
        val high = dominantHue + 13
        if (low >= 0 && high <= 179) {
            Core.inRange(hsv, Scalar(low.toDouble(), 45.0, 45.0), Scalar(high.toDouble(), 255.0, 255.0), mask)
        } else {
            val a = Mat(); val b = Mat()
            if (low < 0) {
                Core.inRange(hsv, Scalar(0.0,45.0,45.0), Scalar(high.toDouble(),255.0,255.0), a)
                Core.inRange(hsv, Scalar((180+low).toDouble(),45.0,45.0), Scalar(179.0,255.0,255.0), b)
            } else {
                Core.inRange(hsv, Scalar(low.toDouble(),45.0,45.0), Scalar(179.0,255.0,255.0), a)
                Core.inRange(hsv, Scalar(0.0,45.0,45.0), Scalar((high-180).toDouble(),255.0,255.0), b)
            }
            Core.bitwise_or(a,b,mask)
            a.release(); b.release()
        }
        hsv.release()

        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(9.0,9.0))
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel)
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel)
        kernel.release()

        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        val copy = mask.clone()
        Imgproc.findContours(copy, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        val minArea = work.cols().toDouble() * work.rows().toDouble() * 0.18
        var best: Rect? = null
        var bestArea = 0.0
        contours.forEach { c ->
            val area = Imgproc.contourArea(c)
            if (area >= minArea) {
                val r = Imgproc.boundingRect(c)
                val aspect = r.width.toDouble() / max(1, r.height).toDouble()
                if (aspect in 1.55..2.45 && area > bestArea) {
                    bestArea = area
                    best = r
                }
            }
            c.release()
        }
        copy.release(); hierarchy.release(); mask.release()
        return best
    }

    private fun detectGuideSegments(tableRgba: Mat): List<GuideSeg> {
        val rgb = Mat(); val hsv = Mat(); val mask = Mat()
        Imgproc.cvtColor(tableRgba, rgb, Imgproc.COLOR_RGBA2RGB)
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
        rgb.release()
        Core.inRange(hsv, Scalar(0.0,0.0,155.0), Scalar(179.0,78.0,255.0), mask)
        hsv.release()

        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0,2.0))
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel)
        kernel.release()
        val border = max(5, (tableRgba.cols()*0.008).toInt())
        Imgproc.rectangle(mask, Point(0.0,0.0), Point((mask.cols()-1).toDouble(),(mask.rows()-1).toDouble()), Scalar(0.0), border)

        val lines = Mat()
        val minLen = max(20.0, tableRgba.cols()*0.022)
        Imgproc.HoughLinesP(mask, lines, 1.0, Math.PI/360.0, 18, minLen, 12.0)
        val out = mutableListOf<GuideSeg>()
        for (row in 0 until lines.rows()) {
            val l = lines.get(row,0) ?: continue
            if (l.size < 4) continue
            val p1 = Vec2(l[0],l[1]); val p2 = Vec2(l[2],l[3])
            val len = (p2-p1).length()
            if (len < minLen) continue
            val mx=(p1.x+p2.x)*0.5; val my=(p1.y+p2.y)*0.5
            val px=tableRgba.cols()*0.015; val py=tableRgba.rows()*0.02
            if (mx<px || mx>tableRgba.cols()-px || my<py || my>tableRgba.rows()-py) continue
            out += GuideSeg(p1,p2,len)
        }
        lines.release(); mask.release()
        return out.sortedByDescending { it.length }.take(100)
    }

    private fun findFork(segments: List<GuideSeg>, tableWidth: Double): Fork? {
        if (segments.isEmpty()) return null
        val longMin = max(55.0, tableWidth*0.07)
        val branchMin = max(18.0, tableWidth*0.018)
        val connectRadius = max(18.0, tableWidth*0.024)
        val lineRadius = max(10.0, tableWidth*0.015)

        var best: Fork? = null
        var bestScore = Double.NEGATIVE_INFINITY

        for (longSeg in segments.filter { it.length >= longMin }.take(24)) {
            val endpoints = listOf(longSeg.p1 to longSeg.p2, longSeg.p2 to longSeg.p1)
            for ((junction, otherEnd) in endpoints) {
                val incoming = (junction-otherEnd).normalized()
                if (incoming.length() < 0.5) continue
                val raw = mutableListOf<Branch>()

                for (seg in segments) {
                    if (seg === longSeg || seg.length < branchMin) continue
                    val d1=(seg.p1-junction).length(); val d2=(seg.p2-junction).length()
                    val nearest=min(d1,d2)
                    if (nearest > connectRadius) continue
                    val lineDist = distancePointToLine(junction, seg.p1, seg.p2)
                    if (lineDist > lineRadius) continue

                    val nearPoint: Vec2
                    val farPoint: Vec2
                    if (d1 <= d2) { nearPoint=seg.p1; farPoint=seg.p2 }
                    else { nearPoint=seg.p2; farPoint=seg.p1 }

                    // Crucial fix: use the segment's own orientation, not junction->farPoint.
                    val direction=(farPoint-nearPoint).normalized()
                    if (direction.length() < 0.5) continue
                    val forward=direction.dot(incoming)
                    if (forward < -0.20) continue
                    val farDistance=(farPoint-junction).length()
                    if (farDistance < branchMin*0.85) continue

                    // Straight-through continuation is VALID and especially important on break shots.
                    val quality=seg.length - nearest*0.30 - lineDist*1.20 + if (forward>0.92) tableWidth*0.012 else 0.0
                    raw += Branch(direction, farPoint, seg.length, quality)
                }

                val branches=deduplicateBranches(raw)
                if (branches.isEmpty()) continue
                val strongest=branches.first().quality
                val accepted=branches.filterIndexed { index, b ->
                    index==0 || b.quality >= max(branchMin*0.75, strongest*0.42)
                }.take(2)
                if (accepted.isEmpty()) continue

                val score=longSeg.length*1.6 + accepted.sumOf { it.quality } + if (accepted.size>=2) tableWidth*0.05 else 0.0
                if (score>bestScore) {
                    bestScore=score
                    best=Fork(junction,incoming,longSeg,accepted)
                }
            }
        }
        return best
    }

    private fun deduplicateBranches(raw: List<Branch>): List<Branch> {
        val kept=mutableListOf<Branch>()
        for (candidate in raw.sortedByDescending { it.quality }) {
            if (kept.none { branchAngle(it.direction,candidate.direction)<8.0 }) kept += candidate
        }
        return kept
    }

    private fun distancePointToLine(p: Vec2, a: Vec2, b: Vec2): Double {
        val ab=b-a
        val len=ab.length()
        if (len<1e-9) return (p-a).length()
        return abs(ab.x*(a.y-p.y) - (a.x-p.x)*ab.y)/len
    }

    private fun extendBranch(startLocal: Vec2, direction0: Vec2, tableRect: Rect, scale: Double, firstKind: SegmentKind, bounces: Int): List<TrajectorySegment> {
        val result=mutableListOf<TrajectorySegment>()
        val radius=tableRect.width*0.012
        val bounds=PlayRect(radius,radius,tableRect.width-radius,tableRect.height-radius)
        var p=startLocal
        var d=direction0.normalized()
        repeat(max(1,bounces+1)) { index ->
            val hit=rayToBoundary(p,d,bounds) ?: return@repeat
            result += TrajectorySegment(
                toGlobal(p,tableRect,scale), toGlobal(hit.first,tableRect,scale),
                if(index==0) firstKind else SegmentKind.BOUNCE,
                firstKind==SegmentKind.CUE_AFTER || index>0
            )
            if(index<bounces) {
                d=Vec2(if(hit.second)-d.x else d.x, if(hit.third)-d.y else d.y)
                p=hit.first+d*0.8
            }
        }
        return result
    }

    private fun rayToBoundary(start: Vec2, d0: Vec2, rect: PlayRect): Triple<Vec2,Boolean,Boolean>? {
        val d=d0.normalized(); val eps=1e-7
        val ts=mutableListOf<Double>()
        if(abs(d.x)>eps) {
            ((rect.left-start.x)/d.x).takeIf{it>eps}?.let(ts::add)
            ((rect.right-start.x)/d.x).takeIf{it>eps}?.let(ts::add)
        }
        if(abs(d.y)>eps) {
            ((rect.top-start.y)/d.y).takeIf{it>eps}?.let(ts::add)
            ((rect.bottom-start.y)/d.y).takeIf{it>eps}?.let(ts::add)
        }
        if(ts.isEmpty()) return null
        val t=ts.min(); val q=start+d*t
        val p=Vec2(q.x.coerceIn(rect.left,rect.right),q.y.coerceIn(rect.top,rect.bottom))
        val hx=abs(p.x-rect.left)<2 || abs(p.x-rect.right)<2
        val hy=abs(p.y-rect.top)<2 || abs(p.y-rect.bottom)<2
        return Triple(p,hx,hy)
    }

    private fun toGlobal(local: Vec2, tableRect: Rect, scale: Double)=Vec2((local.x+tableRect.x)/scale,(local.y+tableRect.y)/scale)

    private fun branchAngle(a0: Vec2,b0: Vec2): Double {
        val a=a0.normalized(); val b=b0.normalized()
        return Math.toDegrees(acos(a.dot(b).coerceIn(-1.0,1.0)))
    }
}
