package com.example.durakassistant.vision

/** A missing or occluded number is unknown, never zero. */
class CountReader(templateText: String) {
    private val templates=templateText.lineSequence().filter{it.isNotBlank()}.map {
        val p=it.split(' ',limit=2);p[0].toInt() to BooleanArray(1280){i->p[1][i]=='1'}
    }.toList()
    fun read(pixels:IntArray,width:Int,height:Int):Int? {
        if(width!=720 || height<790)return null
        val points=mutableListOf<Pair<Int,Int>>()
        for(y in 475 until 555)for(x in 0 until 70){
            val c=pixels[y*width+x];val r=c shr 16 and 255;val g=c shr 8 and 255;val b=c and 255
            if(minOf(r,g,b)>185 && kotlin.math.abs(r-g)<30)points+=x to y
        }
        if(points.size<25)return null
        val x0=points.minOf{it.first};val x1=points.maxOf{it.first}
        val y0=points.minOf{it.second};val y1=points.maxOf{it.second}
        if(y1-y0 !in 20..55 || x1-x0 !in 6..60)return null
        val w=x1-x0+1;val h=y1-y0+1;val source=BooleanArray(w*h)
        for((x,y)in points)source[(y-y0)*w+x-x0]=true
        val a=BooleanArray(1280){i->source[(i/32*h/40)*w+i%32*w/32]}
        val scores=templates.map{(n,t)->
            var inter=0;var total=0
            for(i in a.indices){if(a[i]&&t[i])inter++;if(a[i]||t[i])total++}
            n to if(total==0)0f else inter.toFloat()/total
        }.sortedByDescending{it.second}
        val best=scores.first()
        return best.first.takeIf{best.second>=.66f && best.second-scores[1].second>=.08f}
    }
    fun hasExhaustedDeckIcon(p:IntArray,w:Int,h:Int,core:RecognitionCore):Boolean {
        if(w!=720||h<810)return false
        var white=0;var green=0;var blue=0
        for(y in 580 until 810 step 2)for(x in 0 until 95 step 2){
            val c=p[y*w+x];val r=c shr 16 and 255;val g=c shr 8 and 255;val b=c and 255
            if(minOf(r,g,b)>170)white++
            if(g>130 && g>r*1.08 && g>b*.95)green++
            if(b>r*1.25 && g>r*1.10)blue++
        }
        return white<35 && green<25 && blue>2000 &&
            core.trump(p,w,h,RecognitionCore.Box(0,590,95,810))!=null
    }
}
