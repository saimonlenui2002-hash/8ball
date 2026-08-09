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
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * 4.0 ML Preview.
 * Hough finds only the long native incoming guide. A tiny local ONNX model
 * then confirms the contact ring and native outgoing guide pixels around the
 * endpoint. Final branch angle/end are fitted mathematically from ML pixels.
 */
class NativeGuideMlAnalyzerV40(private val context: Context) {
    private val prefs = context.getSharedPreferences(MainActivity.PREFS, Context.MODE_PRIVATE)
    private val localDnn = LocalGuideDnn(context)

    private data class GuideSeg(val p1: Vec2, val p2: Vec2, val length: Double)
    private data class RayBranch(
        val direction: Vec2,
        val start: Vec2,
        val end: Vec2,
        val score: Double,
        val length: Double
    )
    private data class LocalMlResult(
        val junction: Vec2,
        val branches: List<RayBranch>
    )
    private data class Fork(
        val junction: Vec2,
        val incoming: Vec2,
        val incomingSeg: GuideSeg,
        val branches: List<RayBranch>,
        val usedMl: Boolean
    )

    fun analyze(bitmap: Bitmap): AnalysisResult {
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)
        val scale = min(1.0, 1280.0 / src.cols().toDouble())
        val work = Mat()
        if (scale < 0.999) Imgproc.resize(src, work, Size(), scale, scale, Imgproc.INTER_AREA)
        else src.copyTo(work)
        src.release()

        val tableRect = fixedTable(work)
        if (tableRect == null) {
            work.release()
            return AnalysisResult(bitmap.width, bitmap.height, null, emptyList(), null, emptyList(), null, 0, "ML: кадр слишком мал")
        }

        val table = work.submat(tableRect)
        val whiteMask = buildWhiteMask(table)
        val longSegments = detectLongGuideSegments(whiteMask, table.cols())
        val fork = findFork(table, whiteMask, longSegments, table.cols().toDouble())
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
                val cross = max(6.0, tableRect.width * 0.008)
                out += TrajectorySegment(toGlobal(Vec2(j.x-cross,j.y),tableRect,scale),toGlobal(Vec2(j.x+cross,j.y),tableRect,scale),SegmentKind.AIM)
                out += TrajectorySegment(toGlobal(Vec2(j.x,j.y-cross),tableRect,scale),toGlobal(Vec2(j.x,j.y+cross),tableRect,scale),SegmentKind.AIM)
            }
        } else if (debug) {
            longSegments.take(6).forEach { s ->
                out += TrajectorySegment(toGlobal(s.p1,tableRect,scale),toGlobal(s.p2,tableRect,scale),SegmentKind.AIM)
            }
        }

        whiteMask.release()
        table.release()
        work.release()

        val status = when {
            longSegments.isEmpty() -> "ML: длинная линия не найдена"
            fork == null && localDnn.lastError != null -> "ML недоступна • резерв"
            fork == null -> "ML: ветка не подтверждена"
            fork.usedMl && fork.branches.size == 1 -> "ML: 1 ветка"
            fork.usedMl -> "ML: 2 ветки"
            fork.branches.size == 1 -> "резерв: 1 ветка"
            else -> "резерв: 2 ветки"
        }
        val confidence = when {
            longSegments.isEmpty() -> 20
            fork == null -> 35
            fork.usedMl && fork.branches.size >= 2 -> 99
            fork.usedMl -> 94
            else -> 76
        }
        return AnalysisResult(bitmap.width, bitmap.height, null, emptyList(), null, out, playRect, confidence, status)
    }

    /** Fixed geometry is intentionally not rejected by felt hue anymore. */
    private fun fixedTable(work: Mat): Rect? {
        if (work.cols() < 300 || work.rows() < 200) return null
        val x = (work.cols() * 0.1787).toInt().coerceIn(0, work.cols() - 2)
        val y = (work.rows() * 0.2080).toInt().coerceIn(0, work.rows() - 2)
        val right = (work.cols() * 0.8215).toInt().coerceIn(x + 2, work.cols())
        val bottom = (work.rows() * 0.9410).toInt().coerceIn(y + 2, work.rows())
        return Rect(x, y, right - x, bottom - y)
    }

    private fun buildWhiteMask(table: Mat): Mat {
        val rgb = Mat()
        val hsv = Mat()
        val mask = Mat()
        Imgproc.cvtColor(table, rgb, Imgproc.COLOR_RGBA2RGB)
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
        rgb.release()
        Core.inRange(hsv, Scalar(0.0,0.0,150.0), Scalar(179.0,82.0,255.0), mask)
        hsv.release()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0,2.0))
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel)
        kernel.release()
        val border = max(5, (table.cols() * 0.008).toInt())
        Imgproc.rectangle(mask, Point(0.0,0.0), Point((mask.cols()-1).toDouble(),(mask.rows()-1).toDouble()), Scalar(0.0), border)
        return mask
    }

    private fun detectLongGuideSegments(mask: Mat, tableWidth: Int): List<GuideSeg> {
        val lines = Mat()
        val minLen = max(50.0, tableWidth * 0.065)
        Imgproc.HoughLinesP(mask, lines, 1.0, Math.PI/360.0, 24, minLen, 18.0)
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

    private fun findFork(table: Mat, mask: Mat, segments: List<GuideSeg>, tableWidth: Double): Fork? {
        if (segments.isEmpty()) return null
        var best: Fork? = null
        var bestScore = Double.NEGATIVE_INFINITY

        for (seg in segments.take(14)) {
            for ((seed, otherEnd) in listOf(seg.p1 to seg.p2, seg.p2 to seg.p1)) {
                if (!inside(mask, seed, 4)) continue
                val incomingSeed = (seed-otherEnd).normalized()
                if (incomingSeed.length() < 0.5) continue

                val ml = detectMlBranches(table, seed, incomingSeed, tableWidth)
                val usedMl = ml != null && ml.branches.isNotEmpty()
                val junction = ml?.junction ?: seed
                val incoming = (junction-otherEnd).normalized()
                val branches = if (usedMl) ml!!.branches else scanClassicBranches(mask, junction, incoming, tableWidth)
                if (branches.isEmpty()) continue

                val shiftPenalty = (junction-seed).length() * 0.8
                val score = seg.length * 0.55 + branches.sumOf { it.score } - shiftPenalty +
                    if (usedMl) tableWidth * 0.18 else 0.0 +
                    if (branches.size >= 2) tableWidth * 0.03 else 0.0
                if (score > bestScore) {
                    bestScore = score
                    best = Fork(junction, incoming, seg, branches, usedMl)
                }
            }
        }
        return best
    }

    private fun detectMlBranches(table: Mat, seed: Vec2, incoming: Vec2, tableWidth: Double): LocalMlResult? {
        val cropSide = max(104.0, tableWidth * 0.17).toInt().coerceAtMost(180)
        val pred = localDnn.predict(table, seed, cropSide) ?: return null
        val ring = ringCentroid(pred, seed) ?: return null
        val junction = ring.first
        val ringCount = ring.second
        if (ringCount < 8 || (junction-seed).length() > cropSide * 0.28) return null

        val inner = max(11.0, tableWidth * 0.016)
        val outer = min(cropSide * 0.47, max(48.0, tableWidth * 0.085))
        val minLength = max(11.0, tableWidth * 0.014)
        val raw = mutableListOf<RayBranch>()

        var deg = 0
        while (deg < 360) {
            val a = Math.toRadians(deg.toDouble())
            val dir = Vec2(cos(a), sin(a))
            if (dir.dot(incoming) >= -0.12) {
                traceMlRay(pred, junction, dir, inner, outer)?.let { b ->
                    if (b.length >= minLength) raw += refineMlBranch(pred, junction, b, tableWidth, inner, outer)
                }
            }
            deg += 2
        }

        val kept = mutableListOf<RayBranch>()
        for (candidate in raw.sortedByDescending { it.score }) {
            if (kept.none { branchAngle(it.direction, candidate.direction) < 10.0 }) kept += candidate
            if (kept.size >= 4) break
        }
        if (kept.isEmpty()) return null
        val strongest = kept.first().score
        val accepted = kept.filterIndexed { index, b -> index == 0 || b.score >= strongest * 0.50 }.take(2)
        if (accepted.isEmpty()) return null
        return LocalMlResult(junction, accepted)
    }

    private fun ringCentroid(pred: LocalGuideDnn.Prediction, seed: Vec2): Pair<Vec2,Int>? {
        var sx = 0.0
        var sy = 0.0
        var count = 0
        val maxSeedDistance = pred.side * 0.28
        for (y in 0 until LocalGuideDnn.INPUT) {
            for (x in 0 until LocalGuideDnn.INPUT) {
                if (pred.label(x,y) != 2) continue
                val p = pred.tablePoint(x.toDouble(), y.toDouble())
                if ((p-seed).length() > maxSeedDistance) continue
                sx += p.x
                sy += p.y
                count++
            }
        }
        if (count == 0) return null
        return Vec2(sx/count.toDouble(), sy/count.toDouble()) to count
    }

    private fun traceMlRay(
        pred: LocalGuideDnn.Prediction,
        junction: Vec2,
        dir: Vec2,
        inner: Double,
        outer: Double
    ): RayBranch? {
        val perp = Vec2(-dir.y, dir.x)
        var r = inner
        val step = 1.25
        var started = false
        var first = 0.0
        var last = 0.0
        var hits = 0
        var samples = 0
        var initialMiss = 0
        var missRun = 0
        while (r <= outer) {
            val p = junction + dir*r
            val hit = mlGuideAt(pred, p, perp)
            samples++
            if (hit) {
                if (!started) { started = true; first = r }
                last = r
                hits++
                missRun = 0
            } else if (!started) {
                initialMiss++
                if (initialMiss > 8) return null
            } else {
                missRun++
                if (missRun > 4) break
            }
            r += step
        }
        if (!started || last-first < 7.0) return null
        val length = last-first
        val ratio = hits.toDouble()/max(1,samples).toDouble()
        return RayBranch(dir, junction+dir*first, junction+dir*last, length+ratio*24.0-initialMiss, length)
    }

    private fun mlGuideAt(pred: LocalGuideDnn.Prediction, p: Vec2, perp: Vec2): Boolean {
        var hits = 0
        for (offset in doubleArrayOf(-2.0,-1.0,0.0,1.0,2.0)) {
            val q = p + perp*offset
            val mx = (((q.x-pred.intendedX)/pred.side)*LocalGuideDnn.INPUT).toInt()
            val my = (((q.y-pred.intendedY)/pred.side)*LocalGuideDnn.INPUT).toInt()
            if (pred.label(mx,my) == 1) hits++
        }
        return hits >= 2
    }

    private fun refineMlBranch(
        pred: LocalGuideDnn.Prediction,
        junction: Vec2,
        rough: RayBranch,
        tableWidth: Double,
        inner: Double,
        outer: Double
    ): RayBranch {
        val axis = rough.direction.normalized()
        val corridor = max(4.0, tableWidth * 0.0065)
        val points = mutableListOf<Vec2>()
        for (y in 0 until LocalGuideDnn.INPUT) {
            for (x in 0 until LocalGuideDnn.INPUT) {
                if (pred.label(x,y) != 1) continue
                val p = pred.tablePoint(x.toDouble(), y.toDouble())
                val v = p-junction
                val radial = v.length()
                if (radial < inner*0.65 || radial > outer*1.03) continue
                val projection = v.dot(axis)
                if (projection <= 0.0) continue
                val perpendicular = abs(v.x*axis.y-v.y*axis.x)
                if (perpendicular <= corridor) points += p
            }
        }
        if (points.size < 8) return rough

        val mx = points.sumOf { it.x } / points.size.toDouble()
        val my = points.sumOf { it.y } / points.size.toDouble()
        var xx = 0.0; var xy = 0.0; var yy = 0.0
        for (p in points) {
            val dx=p.x-mx; val dy=p.y-my
            xx += dx*dx; xy += dx*dy; yy += dy*dy
        }
        var angle = 0.5*atan2(2.0*xy, xx-yy)
        var dir = Vec2(cos(angle),sin(angle)).normalized()
        if (dir.dot(axis) < 0.0) dir = dir * -1.0

        val projections = points.map { (it-junction).dot(dir) }.filter { it > 0.0 }.sorted()
        if (projections.size < 8) return rough
        val lo = projections[(projections.size*0.08).toInt().coerceIn(0,projections.lastIndex)]
        val hi = projections[(projections.size*0.96).toInt().coerceIn(0,projections.lastIndex)]
        val length = hi-lo
        if (length < max(9.0, tableWidth*0.012)) return rough

        var residual = 0.0
        for (p in points) {
            val v=p-Vec2(mx,my)
            val d=abs(v.x*dir.y-v.y*dir.x)
            residual += d*d
        }
        val rms=sqrt(residual/points.size.toDouble())
        val score=length+points.size*0.16-rms*1.8
        return RayBranch(dir,junction+dir*lo,junction+dir*hi,score,length)
    }

    /** v3.4 classical local reader kept as a conservative runtime fallback. */
    private fun scanClassicBranches(mask: Mat, junction: Vec2, incoming: Vec2, tableWidth: Double): List<RayBranch> {
        val inner=max(13.0,tableWidth*0.019)
        val outer=max(46.0,tableWidth*0.080)
        val minLength=max(12.0,tableWidth*0.016)
        val raw=mutableListOf<RayBranch>()
        var deg=0
        while(deg<360){
            val a=Math.toRadians(deg.toDouble())
            val dir=Vec2(cos(a),sin(a))
            if(dir.dot(incoming)>=-0.08){
                traceClassicRay(mask,junction,dir,inner,outer)?.let{if(it.length>=minLength)raw+=it}
            }
            deg+=2
        }
        val kept=mutableListOf<RayBranch>()
        for(c in raw.sortedByDescending{it.score}){
            if(kept.none{branchAngle(it.direction,c.direction)<11.0})kept+=c
            if(kept.size>=4)break
        }
        if(kept.isEmpty())return emptyList()
        val strongest=kept.first().score
        return kept.filterIndexed{index,b->index==0||b.score>=strongest*0.55}.take(2)
    }

    private fun traceClassicRay(mask: Mat,junction:Vec2,dir:Vec2,inner:Double,outer:Double):RayBranch?{
        val perp=Vec2(-dir.y,dir.x)
        val step=1.5
        var r=inner;var started=false;var first=0.0;var last=0.0;var hits=0;var samples=0;var missRun=0;var initialMiss=0
        while(r<=outer){
            val p=junction+dir*r
            val white=whiteAt(mask,p,perp)
            samples++
            if(white){if(!started){started=true;first=r};last=r;hits++;missRun=0}
            else if(!started){initialMiss++;if(initialMiss>5)return null}
            else{missRun++;if(missRun>3)break}
            r+=step
        }
        if(!started||last<=first)return null
        val length=last-first
        val ratio=hits.toDouble()/max(1,samples).toDouble()
        return RayBranch(dir,junction+dir*first,junction+dir*last,length+ratio*22.0-initialMiss*1.2,length)
    }

    private fun whiteAt(mask:Mat,p:Vec2,perp:Vec2):Boolean{
        var hits=0
        for(o in doubleArrayOf(-2.0,-1.0,0.0,1.0,2.0)){
            val q=p+perp*o;val x=q.x.toInt();val y=q.y.toInt()
            if(x !in 0 until mask.cols()||y !in 0 until mask.rows())continue
            val v=mask.get(y,x)
            if(v!=null&&v.isNotEmpty()&&v[0]>0.0)hits++
        }
        return hits>=2
    }

    private fun inside(mask:Mat,p:Vec2,margin:Int):Boolean =
        p.x>=margin&&p.y>=margin&&p.x<mask.cols()-margin&&p.y<mask.rows()-margin

    private fun extendBranch(startLocal:Vec2,direction0:Vec2,tableRect:Rect,scale:Double,firstKind:SegmentKind,bounces:Int):List<TrajectorySegment>{
        val result=mutableListOf<TrajectorySegment>()
        val radius=tableRect.width*0.012
        val bounds=PlayRect(radius,radius,tableRect.width-radius,tableRect.height-radius)
        var p=startLocal
        var d=direction0.normalized()
        if(d.length()<0.5)return result
        for(i in 0..bounces){
            val hit=rayToBounds(p,d,bounds)?:break
            val kind=if(i==0)firstKind else SegmentKind.BOUNCE
            result+=TrajectorySegment(toGlobal(p,tableRect,scale),toGlobal(hit,tableRect,scale),kind)
            if(i==bounces)break
            var nx=d.x;var ny=d.y;val eps=2.5
            if(abs(hit.x-bounds.left)<eps||abs(hit.x-bounds.right)<eps)nx=-nx
            if(abs(hit.y-bounds.top)<eps||abs(hit.y-bounds.bottom)<eps)ny=-ny
            d=Vec2(nx,ny).normalized();p=hit+d*1.5
        }
        return result
    }

    private fun rayToBounds(p:Vec2,d:Vec2,b:PlayRect):Vec2?{
        var best=Double.POSITIVE_INFINITY
        fun test(t:Double){
            if(t<=1e-6||t>=best)return
            val x=p.x+d.x*t;val y=p.y+d.y*t
            if(x>=b.left-0.5&&x<=b.right+0.5&&y>=b.top-0.5&&y<=b.bottom+0.5)best=t
        }
        if(abs(d.x)>1e-9){test((b.left-p.x)/d.x);test((b.right-p.x)/d.x)}
        if(abs(d.y)>1e-9){test((b.top-p.y)/d.y);test((b.bottom-p.y)/d.y)}
        return if(best.isFinite())p+d*best else null
    }

    private fun toGlobal(p:Vec2,tableRect:Rect,scale:Double):Vec2=
        Vec2((tableRect.x+p.x)/scale,(tableRect.y+p.y)/scale)

    private fun branchAngle(a:Vec2,b:Vec2):Double{
        val na=a.normalized();val nb=b.normalized()
        return Math.toDegrees(acos(na.dot(nb).coerceIn(-1.0,1.0)))
    }
}
