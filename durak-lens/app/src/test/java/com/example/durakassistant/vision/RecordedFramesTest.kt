package com.example.durakassistant.vision

import java.io.File
import java.io.DataInputStream
import java.io.ByteArrayOutputStream
import java.io.ByteArrayInputStream
import java.util.zip.InflaterInputStream
import org.junit.Assert.*
import org.junit.Test

class RecordedFramesTest {
    private val core=RecognitionCore(File("src/main/assets/glyphs.txt").readText())
    private val counter=CountReader(File("src/main/assets/counts.txt").readText())
    private fun pixels(name:String):IntArray {
        val input=DataInputStream(javaClass.getResourceAsStream("/vision/$name.png"))
        input.use {
            val signature=ByteArray(8);it.readFully(signature)
            val compressed=ByteArrayOutputStream()
            var width=0;var height=0
            while(true){
                val size=it.readInt();val tag=ByteArray(4);it.readFully(tag)
                val data=ByteArray(size);it.readFully(data);it.readInt()
                when(String(tag, Charsets.US_ASCII)){
                    "IHDR" -> {val header=DataInputStream(ByteArrayInputStream(data));width=header.readInt();height=header.readInt()
                        check(header.readUnsignedByte()==8 && header.readUnsignedByte()==2)}
                    "IDAT" -> compressed.write(data)
                    "IEND" -> break
                }
            }
            check(width==720 && height==1574)
            val raw=DataInputStream(InflaterInputStream(ByteArrayInputStream(compressed.toByteArray())))
            val pixels=IntArray(width*height);var previous=IntArray(width*3)
            for(y in 0 until height){
                val filter=raw.readUnsignedByte();val row=IntArray(width*3)
                for(x in row.indices){
                    val a=if(x>=3)row[x-3] else 0;val b=previous[x];val c=if(x>=3)previous[x-3] else 0
                    val predictor=when(filter){
                        0->0;1->a;2->b;3->(a+b)/2
                        4->{val p=a+b-c;val pa=kotlin.math.abs(p-a);val pb=kotlin.math.abs(p-b);val pc=kotlin.math.abs(p-c)
                            if(pa<=pb && pa<=pc)a else if(pb<=pc)b else c}
                        else->error("Unsupported PNG filter")
                    }
                    row[x]=(raw.readUnsignedByte()+predictor) and 255
                }
                for(x in 0 until width)pixels[y*width+x]=0xff000000.toInt() or (row[x*3] shl 16) or (row[x*3+1] shl 8) or row[x*3+2]
                previous=row
            }
            return pixels
        }
    }
    private fun hand(name:String)=core.cards(pixels(name),720,1574,RecognitionCore.Box(0,1040,720,1338),true)
        .map{it.card.toString()}.toSet()
    @Test fun threeCardHandFromFailingRecording(){
        assertEquals(setOf("10♣","10♥","В♠"),hand("07"))
        assertTrue(core.lastScanComplete)
    }
    @Test fun fiveCardHand(){assertEquals(setOf("10♣","Д♣","9♥","10♥","В♠"),hand("02"))}
    @Test fun eightCardFan(){assertEquals(setOf("9♣","10♣","К♣","10♥","К♥","Т♥","К♦","В♠"),hand("11"))}
    @Test fun lateFourCardHand(){assertEquals(setOf("К♥","10♦","В♠","Т♠"),hand("17"))}
    @Test fun lateTwoCardHand(){assertEquals(setOf("К♥","Т♠"),hand("19"))}
    @Test fun readsVisibleDeckNumbers(){
        for((frame,n) in listOf("02" to 12,"07" to 12,"10" to 6,"11" to 4,"12" to 2,"13" to 2)){
            assertEquals("Frame $frame",n,counter.read(pixels(frame),720,1574))
        }
    }
    @Test fun missingNumberIsUnknown(){
        assertNull(counter.read(pixels("17"),720,1574))
        assertNull(counter.read(IntArray(720*1574){0xff365b80.toInt()},720,1574))
        assertNull(counter.read(pixels("04"),720,1574))
    }
    @Test fun emptyDeckNeedsVisibleTrumpIcon(){
        assertTrue(counter.hasExhaustedDeckIcon(pixels("17"),720,1574,core))
        assertFalse(counter.hasExhaustedDeckIcon(pixels("02"),720,1574,core))
    }
    @Test fun seesSixCardsOnCoveredTable(){
        val cards=core.cards(pixels("07"),720,1574,RecognitionCore.Box(80,470,680,790),false).map{it.card.toString()}.toSet()
        assertEquals(setOf("9♦","Д♦","9♥","В♥","Д♣","К♠"),cards)
    }
}
