package com.example.durakassistant.vision

import java.io.File
import javax.imageio.ImageIO
import org.junit.Assert.*
import org.junit.Test

class RecordedFramesTest {
    private val core=RecognitionCore(File("src/main/assets/glyphs.txt").readText())
    private val counter=CountReader(File("src/main/assets/counts.txt").readText())
    private fun pixels(name:String):IntArray {
        val image=ImageIO.read(javaClass.getResourceAsStream("/vision/$name.png"))
        return image.getRGB(0,0,image.width,image.height,null,0,image.width)
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
