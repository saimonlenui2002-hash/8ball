package com.example.durakassistant.game

data class TablePair(val attack: Card, val defense: Card? = null)

enum class TurnPhase { ATTACK, DEFEND, WAIT, UNKNOWN }

data class Observation(
    val hand: Set<Card>,
    val table: List<TablePair>,
    val trump: Suit?,
    val deckCount: Int?,
    val opponentCount: Int?,
    val phase: TurnPhase = TurnPhase.UNKNOWN,
    val confidence: Float = 0f
) {
    val tableCards: Set<Card> get() = table.flatMap { listOfNotNull(it.attack, it.defense) }.toSet()
}

data class GameKnowledge(
    val hand: Set<Card> = emptySet(),
    val table: List<TablePair> = emptyList(),
    val trump: Suit? = null,
    val deckCount: Int = 12,
    val opponentCount: Int = 6,
    val knownOpponent: Set<Card> = emptySet(),
    val possibleOpponent: Set<Card> = Deck24.cards,
    val discarded: Set<Card> = emptySet(),
    val opponentTaken: Set<Card> = emptySet(),
    val lastEvent: String = "Ожидаю начало партии",
    val phase: TurnPhase = TurnPhase.UNKNOWN,
    val confidence: Float = 0f
)
