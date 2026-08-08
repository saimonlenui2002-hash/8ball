package com.example.pooltrajectory

import kotlin.math.hypot

data class Vec2(val x: Double, val y: Double) {
    operator fun plus(o: Vec2) = Vec2(x + o.x, y + o.y)
    operator fun minus(o: Vec2) = Vec2(x - o.x, y - o.y)
    operator fun times(k: Double) = Vec2(x * k, y * k)
    fun dot(o: Vec2) = x * o.x + y * o.y
    fun length() = hypot(x, y)
    fun normalized(): Vec2 { val l = length(); return if (l < 1e-9) Vec2(0.0,0.0) else Vec2(x/l,y/l) }
}

data class Ball(val center: Vec2, val radius: Double, val isCue: Boolean = false)

data class PlayRect(val left: Double, val top: Double, val right: Double, val bottom: Double) {
    fun inset(v: Double) = PlayRect(left+v, top+v, right-v, bottom-v)
}

enum class SegmentKind { AIM, OBJECT, CUE_AFTER, BOUNCE }

data class TrajectorySegment(val start: Vec2, val end: Vec2, val kind: SegmentKind, val dashed: Boolean = false)

data class AnalysisResult(
    val frameWidth: Int,
    val frameHeight: Int,
    val cueBall: Ball?,
    val balls: List<Ball>,
    val ghostCueCenter: Vec2?,
    val segments: List<TrajectorySegment>,
    val playRect: PlayRect?,
    val confidence: Int,
    val status: String
)
