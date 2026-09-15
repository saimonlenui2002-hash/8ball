package com.example.durakassistant.vision

import com.example.durakassistant.game.*
import kotlin.math.*

/** Android-independent pixel reader, also exercised against recorded-frame fixtures. */
class RecognitionCore(templateText: String) {
    data class Box(val left: Int, val top: Int, val right: Int, val bottom: Int) {
        val width get() = right - left
        val height get() = bottom - top
        val cx get() = (left + right) / 2
    }
    data class Detection(val card: Card, val box: Box, val score: Float)
    private data class Part(val points: List<Int>, val box: Box, val red: Boolean)
    private val templates = templateText.lineSequence().filter { it.isNotBlank() }.map {
        val pair = it.split(' ', limit = 2)
        pair[0] to BooleanArray(28 * 44) { i -> pair[1][i] == '1' }
    }.toList()
    private val ranks = templates.filter { it.first in Rank.entries.map { rank -> rank.name } }
    private val suits = templates.filter { it.first in Suit.entries.map { suit -> suit.name } }

    var lastScanComplete: Boolean = false
        private set

    fun cards(pixels: IntArray, width: Int, height: Int, region: Box, hand: Boolean): List<Detection> {
        val parts = components(pixels, width, height, region)
        val rankParts = parts.filter {
            it.box.height in (if (hand) 65..135 else 28..65) &&
                it.box.top <= (if (hand) 1210 else 950) && it.box.width <= it.box.height * .85
        }
        val merged = mutableListOf<Part>()
        for (a in rankParts) for (b in rankParts) {
            val gap = b.box.left - a.box.right
            if (a === b || a.red != b.red || b.box.left <= a.box.left) continue
            if (gap !in (if (hand) -12..15 else -7..9)) continue
            if (abs(a.box.top - b.box.top) > (if (hand) 15 else 9)) continue
            if (min(a.box.height, b.box.height) < max(a.box.height, b.box.height) * .8) continue
            merged += combine(listOf(a, b), width)
        }
        val output = mutableListOf<Detection>()
        val anchors = mutableListOf<Box>()
        for (rankPart in rankParts + merged) {
            val rb = rankPart.box
            if (!whiteSurround(pixels, width, height, rb)) continue
            val rank = best(normalize(rankPart, width), ranks) ?: continue
            if (rank.second < .60f) continue
            anchors += rb
            val nearby = parts.filter { s ->
                s.red == rankPart.red && s.box.top >= rb.bottom - rb.height * .03 &&
                    s.box.top <= rb.bottom + rb.height * .32 &&
                    abs(s.box.cx - rb.cx) <= rb.height * .4 &&
                    s.box.height > rb.height * .25 && s.box.height < rb.height * .85
            }.toMutableList()
            // Clubs in the small table font can consist of three disjoint lobes.
            val fragments = parts.filter { s ->
                s.red == rankPart.red && s.box.top >= rb.bottom &&
                    s.box.bottom <= rb.bottom + rb.height * .85 &&
                    s.box.left >= rb.cx - rb.height * .4 && s.box.right <= rb.cx + rb.height * .4
            }
            if (fragments.size in 2..4) nearby += combine(fragments, width)
            val suited = suits.filter { (name, _) -> Suit.valueOf(name).red == rankPart.red }
            val suitMatches = nearby.mapNotNull { part ->
                val match = best(normalize(part, width), suited) ?: return@mapNotNull null
                if (match.second < .60f) null else Triple(part, match.first, match.second)
            }
            val suit = suitMatches.maxByOrNull { it.third } ?: continue
            val box = Box(min(rb.left, suit.first.box.left), rb.top,
                max(rb.right, suit.first.box.right), suit.first.box.bottom)
            output += Detection(Card(Rank.valueOf(rank.first), Suit.valueOf(suit.second)), box,
                (rank.second + suit.third) / 2)
        }
        val kept = mutableListOf<Detection>()
        for (d in output.sortedByDescending { it.score }) {
            if (kept.none { abs(it.box.cx - d.box.cx) < (if (hand) 30 else 20) &&
                    abs(it.box.top - d.box.top) < (if (hand) 100 else 55) }) kept += d
        }
        val spots=mutableListOf<Box>()
        for(b in anchors.sortedByDescending{it.width}) {
            if(spots.none{abs(it.cx-b.cx)<(if(hand)30 else 20) && abs(it.top-b.top)<(if(hand)100 else 55)})spots+=b
        }
        lastScanComplete = kept.size == spots.size && kept.map{it.card}.toSet().size==kept.size
        return kept.sortedBy { it.box.left }.distinctBy { it.card }
    }

    fun trump(pixels: IntArray, width: Int, height: Int, region: Box): Suit? {
        return components(pixels, width, height, region).filter { it.box.height in 15..100 }
            .mapNotNull { part ->
                best(normalize(part, width), suits.filter { Suit.valueOf(it.first).red == part.red })
            }.maxByOrNull { it.second }?.takeIf { it.second >= .72f }?.let { Suit.valueOf(it.first) }
    }

    private fun whiteSurround(pixels: IntArray, w: Int, h: Int, b: Box): Boolean {
        var white = 0
        for (x in listOf(b.left - 2, b.right + 2)) for (k in 1..3) {
            val y = b.top + b.height * k / 4
            if (x !in 0 until w || y !in 0 until h) continue
            val c = pixels[y*w+x]; val r = c shr 16 and 255; val g = c shr 8 and 255; val blue = c and 255
            if (minOf(r,g,blue) > 160 && maxOf(r,g,blue)-minOf(r,g,blue)<45) white++
        }
        return white >= 3
    }

    private fun components(p: IntArray, w: Int, h: Int, region: Box): List<Part> {
        val left = region.left.coerceIn(0,w); val right = region.right.coerceIn(left,w)
        val top = region.top.coerceIn(0,h); val bottom = region.bottom.coerceIn(top,h)
        val rw = right-left; val rh = bottom-top
        if (rw == 0 || rh == 0) return emptyList()
        val fg = BooleanArray(rw*rh) { ink(p[(top+it/rw)*w+left+it%rw]) }
        val seen = BooleanArray(fg.size); val queue = IntArray(fg.size); val out = mutableListOf<Part>()
        for (start in fg.indices) {
            if (seen[start] || !fg[start]) continue
            var head=0; var tail=1; queue[0]=start; seen[start]=true
            val points=ArrayList<Int>(); var red=0
            var x0=w; var x1=0; var y0=h; var y1=0
            while(head<tail) {
                val i=queue[head++]; val x=left+i%rw; val y=top+i/rw
                val index=y*w+x; points+=index; if(isRed(p[index])) red++
                x0=min(x0,x); x1=max(x1,x); y0=min(y0,y); y1=max(y1,y)
                fun add(n:Int) { if (!seen[n] && fg[n]) {seen[n]=true; queue[tail++]=n} }
                if(i%rw>0)add(i-1); if(i%rw+1<rw)add(i+1)
                if(i>=rw)add(i-rw); if(i+rw<fg.size)add(i+rw)
            }
            if(points.size>=18)out+=Part(points,Box(x0,y0,x1+1,y1+1),red>points.size/2)
        }
        return out
    }

    private fun combine(parts: List<Part>, width: Int): Part {
        val points=parts.flatMap { it.points }.distinct()
        return Part(points,Box(points.minOf{it%width},points.minOf{it/width},
            points.maxOf{it%width}+1,points.maxOf{it/width}+1),parts.first().red)
    }

    private fun normalize(part: Part, width: Int): BooleanArray {
        val b=part.box; val source=BooleanArray(b.width*b.height)
        for(p in part.points)source[(p/width-b.top)*b.width+p%width-b.left]=true
        // Sample every destination pixel. Forward point mapping left holes when
        // a small glyph was enlarged and changed scores with screen resolution.
        return BooleanArray(28*44) { i -> source[(i/28*b.height/44)*b.width+i%28*b.width/28] }
    }

    private fun best(a: BooleanArray, options: List<Pair<String,BooleanArray>>): Pair<String,Float>? {
        var bestName=""; var bestScore=0f
        for ((name,t) in options) for(dy in -1..1) for(dx in -1..1) {
            var intersection=0; var union=0
            for(y in 0 until 44) for(x in 0 until 28) {
                val av=a[y*28+x]; val xx=x+dx; val yy=y+dy
                val bv=xx in 0..27 && yy in 0..43 && t[yy*28+xx]
                if(av||bv)union++; if(av&&bv)intersection++
            }
            val score=if(union==0)0f else intersection.toFloat()/union
            if(score>bestScore){bestScore=score;bestName=name}
        }
        return if(bestName.isEmpty())null else bestName to bestScore
    }

    private fun ink(c:Int):Boolean = ((c shr 16 and 255)<105 && (c shr 8 and 255)<105 && (c and 255)<105)||isRed(c)
    private fun isRed(c:Int):Boolean {val r=c shr 16 and 255;return r>145 && r>(c shr 8 and 255)*1.45 && r>(c and 255)*1.25}
}
