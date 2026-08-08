package com.example.pooltrajectory

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

object PhysicsEngine {
    data class Collision(val ball: Ball, val t: Double, val ghostCenter: Vec2)

    fun nearestBallCollision(cue: Ball, direction: Vec2, balls: List<Ball>): Collision? {
        val d = direction.normalized()
        var best: Collision? = null
        for (ball in balls) {
            if (ball.isCue) continue
            val f = cue.center - ball.center
            val r = cue.radius + ball.radius
            val b = 2.0 * d.dot(f)
            val c = f.dot(f) - r * r
            val disc = b*b - 4.0*c
            if (disc < 0.0) continue
            val root = sqrt(disc)
            val ts = listOf((-b-root)/2.0, (-b+root)/2.0).filter { it > 0.0 }
            if (ts.isEmpty()) continue
            val t = ts.min()
            if (best == null || t < best.t) best = Collision(ball, t, cue.center + d*t)
        }
        return best
    }

    fun solve(cue: Ball, aim: Vec2, balls: List<Ball>, playRect: PlayRect, maxBounces: Int): Pair<Vec2?, List<TrajectorySegment>> {
        val d = aim.normalized()
        val collision = nearestBallCollision(cue, d, balls)
        if (collision == null) {
            val hit = rayToBoundary(cue.center, d, playRect.inset(cue.radius)) ?: return null to emptyList()
            return null to listOf(TrajectorySegment(cue.center, hit.first, SegmentKind.AIM))
        }
        val ghost = collision.ghostCenter
        val target = collision.ball
        val normal = (target.center - ghost).normalized()
        val cueResidual = d - normal * max(0.0, d.dot(normal))
        val out = mutableListOf(TrajectorySegment(cue.center, ghost, SegmentKind.AIM))
        out += reflectedPath(target.center, normal, playRect.inset(target.radius), maxBounces)
        if (cueResidual.length() > 0.08) {
            rayToBoundary(ghost, cueResidual.normalized(), playRect.inset(cue.radius))?.let {
                out += TrajectorySegment(ghost, it.first, SegmentKind.CUE_AFTER, true)
            }
        }
        return ghost to out
    }

    private fun reflectedPath(start: Vec2, direction: Vec2, rect: PlayRect, bounces: Int): List<TrajectorySegment> {
        val result = mutableListOf<TrajectorySegment>()
        var p = start
        var d = direction.normalized()
        repeat(max(1, bounces + 1)) { i ->
            val hit = rayToBoundary(p, d, rect) ?: return@repeat
            result += TrajectorySegment(p, hit.first, if (i == 0) SegmentKind.OBJECT else SegmentKind.BOUNCE, i > 0)
            if (i < bounces) {
                val hx = hit.second
                val hy = hit.third
                d = Vec2(if (hx) -d.x else d.x, if (hy) -d.y else d.y)
                p = hit.first + d * 0.75
            }
        }
        return result
    }

    private fun rayToBoundary(start: Vec2, d0: Vec2, rect: PlayRect): Triple<Vec2, Boolean, Boolean>? {
        val d = d0.normalized(); val eps = 1e-7
        val ts = mutableListOf<Double>()
        if (abs(d.x) > eps) { (rect.left-start.x).div(d.x).takeIf { it > eps }?.let(ts::add); (rect.right-start.x).div(d.x).takeIf { it > eps }?.let(ts::add) }
        if (abs(d.y) > eps) { (rect.top-start.y).div(d.y).takeIf { it > eps }?.let(ts::add); (rect.bottom-start.y).div(d.y).takeIf { it > eps }?.let(ts::add) }
        if (ts.isEmpty()) return null
        val t = ts.min(); val q = start + d*t
        val p = Vec2(min(rect.right,max(rect.left,q.x)), min(rect.bottom,max(rect.top,q.y)))
        val hx = abs(p.x-rect.left)<2 || abs(p.x-rect.right)<2
        val hy = abs(p.y-rect.top)<2 || abs(p.y-rect.bottom)<2
        return Triple(p,hx,hy)
    }
}
