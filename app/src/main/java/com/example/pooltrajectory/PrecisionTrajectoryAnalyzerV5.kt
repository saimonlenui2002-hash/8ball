package com.example.pooltrajectory

import android.content.Context
import android.graphics.Bitmap
import kotlin.math.abs

/**
 * 5.0 precision layer.
 *
 * The 4.2.1 detector remains responsible for reading the game's own short
 * object/cue branches. 5.0 treats those measured branch directions as the
 * authoritative signal, reconstructs their common collision junction and then
 * ray-casts both paths against an effective cushion rectangle inset by one ball
 * radius. This avoids extrapolating to the visible felt edge and makes every
 * bank use the centre-of-ball geometry.
 */
class PrecisionTrajectoryAnalyzerV5(private val context: Context) {
    private val detector = NativeGuideMlAnalyzerV421(context)
    private val prefs = context.getSharedPreferences(MainActivity.PREFS, Context.MODE_PRIVATE)

    private data class Hit(val point: Vec2, val vertical: Boolean, val horizontal: Boolean)

    fun analyze(bitmap: Bitmap): AnalysisResult {
        val raw = detector.analyze(bitmap)
        val play = raw.playRect
            ?: return raw.copy(status = raw.status.replace("ML 4.2.1", "ML 5.0"))

        val objectSeed = raw.segments.firstOrNull { it.kind == SegmentKind.OBJECT }
        val cueSeed = raw.segments.firstOrNull { it.kind == SegmentKind.CUE_AFTER }

        if (objectSeed == null && cueSeed == null) {
            return raw.copy(status = raw.status.replace("ML 4.2.1", "ML 5.0"))
        }

        val objectDir = objectSeed?.let { (it.end - it.start).normalized() }
        val cueDir = cueSeed?.let { (it.end - it.start).normalized() }

        val junction = when {
            objectSeed != null && cueSeed != null && objectDir != null && cueDir != null -> {
                stableIntersection(objectSeed.start, objectDir, cueSeed.start, cueDir, play)
            }
            objectSeed != null -> objectSeed.start
            else -> cueSeed!!.start
        }

        val tableWidth = (play.right - play.left).coerceAtLeast(1.0)
        // Initial calibration prior for the current 8 Ball Pool table layout.
        // It is intentionally kept in table-relative units rather than screen px.
        val radius = (tableWidth * 0.0142).coerceIn(8.0, 34.0)
        val effective = play.inset(radius)
        val bounces = prefs.getInt(MainActivity.KEY_BOUNCES, 0).coerceIn(0, 2)

        val rebuilt = mutableListOf<TrajectorySegment>()

        // Keep detector AIM diagnostics only when they exist (debug mode).
        raw.segments.filter { it.kind == SegmentKind.AIM }.forEach { rebuilt += it }

        if (objectDir != null && objectDir.length() > 0.5) {
            rebuilt += traceToCushions(junction, objectDir, effective, SegmentKind.OBJECT, bounces)
        }
        if (cueDir != null && cueDir.length() > 0.5) {
            rebuilt += traceToCushions(junction, cueDir, effective, SegmentKind.CUE_AFTER, bounces)
        }

        if (rebuilt.none { it.kind != SegmentKind.AIM }) {
            return raw.copy(status = "ML 5.0: ветки распознаны недостаточно уверенно")
        }

        val both = objectDir != null && cueDir != null
        val confidence = (raw.confidence + if (both) 3 else 0).coerceIn(0, 99)
        val status = when {
            both -> "ML 5.0: точная геометрия • объект + биток"
            objectDir != null -> "ML 5.0: точная геометрия • прицельный шар"
            else -> "ML 5.0: точная геометрия • биток"
        }

        return raw.copy(
            ghostCueCenter = junction,
            segments = rebuilt,
            confidence = confidence,
            status = status
        )
    }

    private fun stableIntersection(
        p: Vec2,
        r: Vec2,
        q: Vec2,
        s: Vec2,
        play: PlayRect
    ): Vec2 {
        val cross = cross(r, s)
        val fallback = Vec2((p.x + q.x) * 0.5, (p.y + q.y) * 0.5)
        if (abs(cross) < 0.035) return clamp(fallback, play)

        val t = cross(q - p, s) / cross
        val candidate = p + r * t
        val width = (play.right - play.left).coerceAtLeast(1.0)
        val tooFar = (candidate - p).length() > width * 0.20 ||
            (candidate - q).length() > width * 0.20

        return clamp(if (tooFar) fallback else candidate, play)
    }

    private fun traceToCushions(
        start: Vec2,
        initialDirection: Vec2,
        rect: PlayRect,
        firstKind: SegmentKind,
        bounceCount: Int
    ): List<TrajectorySegment> {
        val out = mutableListOf<TrajectorySegment>()
        var drawStart = clamp(start, rect)
        var rayStart = drawStart
        var dir = initialDirection.normalized()

        for (leg in 0..bounceCount) {
            val hit = hitRect(rayStart, dir, rect) ?: break
            if ((hit.point - drawStart).length() < 2.0) break

            out += TrajectorySegment(
                drawStart,
                hit.point,
                if (leg == 0) firstKind else SegmentKind.BOUNCE
            )

            if (leg == bounceCount) break

            dir = Vec2(
                if (hit.vertical) -dir.x else dir.x,
                if (hit.horizontal) -dir.y else dir.y
            ).normalized()

            drawStart = hit.point
            // Move only the ray origin, not the rendered line start, to avoid
            // immediately re-hitting the same cushion due to floating point noise.
            rayStart = hit.point + dir * 0.75
        }
        return out
    }

    private fun hitRect(p: Vec2, d: Vec2, rect: PlayRect): Hit? {
        val eps = 1e-7
        var bestT = Double.POSITIVE_INFINITY

        fun consider(t: Double) {
            if (t > 1e-4 && t < bestT) bestT = t
        }

        if (d.x > eps) consider((rect.right - p.x) / d.x)
        else if (d.x < -eps) consider((rect.left - p.x) / d.x)

        if (d.y > eps) consider((rect.bottom - p.y) / d.y)
        else if (d.y < -eps) consider((rect.top - p.y) / d.y)

        if (!bestT.isFinite()) return null
        val hit = p + d * bestT
        if (hit.x < rect.left - 1.5 || hit.x > rect.right + 1.5 ||
            hit.y < rect.top - 1.5 || hit.y > rect.bottom + 1.5) return null

        val vertical = abs(hit.x - rect.left) <= 1.5 || abs(hit.x - rect.right) <= 1.5
        val horizontal = abs(hit.y - rect.top) <= 1.5 || abs(hit.y - rect.bottom) <= 1.5
        if (!vertical && !horizontal) return null

        return Hit(clamp(hit, rect), vertical, horizontal)
    }

    private fun clamp(p: Vec2, rect: PlayRect) = Vec2(
        p.x.coerceIn(rect.left, rect.right),
        p.y.coerceIn(rect.top, rect.bottom)
    )

    private fun cross(a: Vec2, b: Vec2) = a.x * b.y - a.y * b.x
}
