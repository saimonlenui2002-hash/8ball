package com.example.pooltrajectory

import android.content.Context
import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

class FrameAnalyzer(private val context: Context) {
    private val prefs = context.getSharedPreferences(MainActivity.PREFS, Context.MODE_PRIVATE)

    fun analyze(bitmap: Bitmap): AnalysisResult {
        val src = Mat(); Utils.bitmapToMat(bitmap, src)
        val scale = min(1.0, 960.0 / src.cols().toDouble())
        val work = Mat()
        if (scale < 0.999) Imgproc.resize(src, work, Size(), scale, scale, Imgproc.INTER_AREA) else src.copyTo(work)
        src.release()
        val gray = Mat(); Imgproc.cvtColor(work, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.GaussianBlur(gray, gray, Size(7.0,7.0), 1.5)

        val circles = Mat()
        val sensitivity = prefs.getInt(MainActivity.KEY_SENSITIVITY,22).coerceIn(12,38)
        val minR = max(7, (work.cols() * 0.008).toInt())
        val maxR = max(minR + 3, (work.cols() * 0.026).toInt())
        Imgproc.HoughCircles(gray, circles, Imgproc.HOUGH_GRADIENT, 1.2, minR * 1.8, 110.0, sensitivity.toDouble(), minR, maxR)

        val balls = mutableListOf<Ball>()
        if (circles.rows() > 0) {
            for (i in 0 until circles.cols()) {
                val c = circles.get(0,i) ?: continue
                if (c.size < 3) continue
                balls += Ball(Vec2(c[0]/scale,c[1]/scale), c[2]/scale, false)
            }
        }
        circles.release()

        val cueIndex = findCueBall(bitmap, balls)
        val labeled = balls.mapIndexed { i,b -> b.copy(isCue = i == cueIndex) }
        val cue = labeled.getOrNull(cueIndex)
        val rect = PlayRect(bitmap.width*0.04, bitmap.height*0.12, bitmap.width*0.96, bitmap.height*0.88)
        var ghost: Vec2? = null
        var segments = emptyList<TrajectorySegment>()

        if (cue != null && labeled.size > 1) {
            val aim = detectAim(gray, cue, scale) ?: run {
                val nearest = labeled.filterNot { it.isCue }.minByOrNull { (it.center-cue.center).length() }
                if (nearest != null) (nearest.center-cue.center).normalized() else Vec2(1.0,0.0)
            }
            val solved = PhysicsEngine.solve(cue, aim, labeled, rect, prefs.getInt(MainActivity.KEY_BOUNCES,1).coerceIn(0,2))
            ghost = solved.first; segments = solved.second
        }

        gray.release(); work.release()
        val confidence = when { cue == null -> 20; labeled.size >= 4 -> 80; else -> 55 }
        val status = when { cue == null -> "биток не найден"; segments.isEmpty() -> "ищу направление"; else -> "траектория" }
        return AnalysisResult(bitmap.width, bitmap.height, cue, labeled, ghost, segments, rect, confidence, status)
    }

    private fun findCueBall(bitmap: Bitmap, balls: List<Ball>): Int {
        if (balls.isEmpty()) return -1
        return balls.indices.maxByOrNull { i ->
            val b = balls[i]
            val x = b.center.x.toInt().coerceIn(0, bitmap.width-1)
            val y = b.center.y.toInt().coerceIn(0, bitmap.height-1)
            val p = bitmap.getPixel(x,y)
            android.graphics.Color.red(p)+android.graphics.Color.green(p)+android.graphics.Color.blue(p)
        } ?: -1
    }

    private fun detectAim(gray: Mat, cue: Ball, scale: Double): Vec2? {
        val edges = Mat(); Imgproc.Canny(gray, edges, 70.0, 160.0)
        val lines = Mat(); Imgproc.HoughLinesP(edges, lines, 1.0, Math.PI/180.0, 55, 65.0, 18.0)
        edges.release()
        val cx = cue.center.x*scale; val cy = cue.center.y*scale
        var best: Vec2? = null; var bestDist = Double.MAX_VALUE
        for (r in 0 until lines.rows()) {
            val l = lines.get(r,0) ?: continue
            if (l.size < 4) continue
            val x1=l[0]; val y1=l[1]; val x2=l[2]; val y2=l[3]
            val dx=x2-x1; val dy=y2-y1; val len=hypot(dx,dy); if (len < 40) continue
            val t=((cx-x1)*dx+(cy-y1)*dy)/(len*len)
            val px=x1+t.coerceIn(0.0,1.0)*dx; val py=y1+t.coerceIn(0.0,1.0)*dy
            val dist=hypot(px-cx,py-cy)
            if (dist < bestDist && dist < cue.radius*scale*3.0) {
                bestDist=dist
                val d1=Vec2(dx,dy).normalized(); val mid=Vec2((x1+x2)/2.0,(y1+y2)/2.0)
                best = if ((mid-Vec2(cx,cy)).dot(d1) >= 0) d1 else d1*-1.0
            }
        }
        lines.release(); return best
    }
}
