package com.example.durakassistant.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import com.example.durakassistant.game.*
import kotlin.math.abs

class FrameAnalyzer(context: Context) {
    private val core=RecognitionCore(context.assets.open("glyphs.txt").bufferedReader().use{it.readText()})
    private val counts=CountReader(context.assets.open("counts.txt").bufferedReader().use{it.readText()})

    fun analyze(frame:Bitmap):Observation {
        val canonical=canonicalFrame(frame)
        try {
            val p=IntArray(720*1574);canonical.getPixels(p,0,720,0,0,720,1574)
            val hand=core.cards(p,720,1574,RecognitionCore.Box(0,1040,720,1338),true)
            val handComplete=core.lastScanComplete && hand.isNotEmpty()
            val detections=core.cards(p,720,1574,RecognitionCore.Box(80,470,680,790),false)
            val tableComplete=core.lastScanComplete
            val deck=counts.read(p,720,1574) ?: if(counts.hasExhaustedDeckIcon(p,720,1574,core))0 else null
            val trump=core.trump(p,720,1574,RecognitionCore.Box(0,620,90,810))
            val table=pairTable(detections)
            // The green border around the player's avatar is the turn signal.
            // Red buttons are available while waiting too, so they are not used.
            var green=0
            for(y in 1360 until 1480 step 3)for(x in 295 until 425 step 3){
                val c=p[y*720+x];val r=c shr 16 and 255;val g=c shr 8 and 255;val b=c and 255
                if(g>140 && r in 60..190 && g>r*1.15 && b<100)green++
            }
            val phase=when {
                green<45 -> TurnPhase.WAIT
                !handComplete || !tableComplete -> TurnPhase.UNKNOWN
                table.isEmpty() -> TurnPhase.ATTACK
                table.any{it.defense==null} -> TurnPhase.DEFEND
                else -> TurnPhase.ATTACK
            }
            return Observation(hand.map{it.card}.toSet(),table,trump,deck,null,phase,
                (hand+detections).map{it.score}.takeIf{it.isNotEmpty()}?.average()?.toFloat()?:0f,
                handComplete=handComplete,tableComplete=tableComplete)
        } finally {canonical.recycle()}
    }

    private fun canonicalFrame(source:Bitmap):Bitmap {
        val scaled=Bitmap.createScaledBitmap(source,720,(source.height*720f/source.width).toInt(),true)
        val w=720;val h=scaled.height;val pixels=IntArray(w*h);scaled.getPixels(pixels,0,w,0,0,w,h)
        // Align to the game board and the white action strip, not the status bar.
        fun blue(y:Int):Boolean {
            var n=0
            for(x in 100 until 700 step 10){val c=pixels[y*w+x];val r=c shr 16 and 255;val g=c shr 8 and 255;val b=c and 255
                if(b>r*1.25 && g>r*1.1 && r in 20..130)n++}
            return n>38
        }
        val top=(0 until h/3).firstOrNull{blue(it)}?:78
        val bottom=(h*3/4 until h-20).firstOrNull{y ->
            var white=0
            for(x in 10 until 710 step 10){val c=pixels[y*w+x];if((c shr 16 and 255)>220&&(c shr 8 and 255)>220&&(c and 255)>220)white++}
            white>66
        }?: (h*1338/1574)
        val out=Bitmap.createBitmap(720,1574,Bitmap.Config.ARGB_8888)
        val sy=(1338f-78)/((bottom-top).coerceAtLeast(500))
        val dest=Rect(0,(78-top*sy).toInt(),720,(78+(h-top)*sy).toInt())
        Canvas(out).drawBitmap(scaled,null,dest,null)
        if(scaled!==source)scaled.recycle()
        return out
    }

    private fun pairTable(cards:List<RecognitionCore.Detection>):List<TablePair> {
        val remaining=cards.sortedBy{it.box.top}.toMutableList();val pairs=mutableListOf<TablePair>()
        while(remaining.isNotEmpty()) {
            val attack=remaining.removeAt(0)
            val defense=remaining.filter{
                it.box.left>attack.box.left && it.box.left-attack.box.left in 15..85 &&
                    it.box.top-attack.box.top in 5..55 && it.card!=attack.card
            }.minByOrNull{abs(it.box.top-attack.box.top)+abs(it.box.left-attack.box.left)}
            if(defense!=null)remaining.remove(defense)
            pairs+=TablePair(attack.card,defense?.card)
        }
        return pairs
    }
}
