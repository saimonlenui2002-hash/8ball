package com.example.durakassistant.game

class GameTracker {
    private var state = GameKnowledge()

    @Synchronized
    fun reset() {
        state = GameKnowledge()
    }

    @Synchronized
    fun current(): GameKnowledge = state

    @Synchronized
    fun accept(observation: Observation): GameKnowledge {
        val previous = state
        val previousTable = previous.table.flatMap { listOfNotNull(it.attack, it.defense) }.toSet()
        val currentTable = observation.tableCards
        val observedDeck = observation.deckCount ?: previous.deckCount
        var discarded = previous.discarded
        var knownOpponent = previous.knownOpponent

        // A cleared, fully covered table is beaten. An uncovered table was taken
        // by one of the players. This is more reliable than comparing deck OCR,
        // which can briefly fluctuate during the dealing animation.
        if (previousTable.isNotEmpty() && currentTable.isEmpty()) {
            val playerTook = observation.hand.any { it in previousTable }
            val fullyCovered = previous.table.all { it.defense != null }
            if (!playerTook && !fullyCovered) {
                knownOpponent = knownOpponent + previousTable
            } else if (!playerTook && fullyCovered) {
                discarded = discarded + previousTable
            }
        }

        // Once an already known card appears on the table, it has left the hand.
        knownOpponent = knownOpponent - currentTable
        discarded = discarded - currentTable

        val hand = stableSet(observation.hand, previous.hand, observation.confidence)
        val trump = observation.trump ?: previous.trump
        // The deck cannot grow during one game. The stabilizer already rejects
        // single-frame OCR mistakes; this guard rejects any remaining increase.
        val deckCount = minOf(previous.deckCount, observedDeck)
        var opponentCount = observation.opponentCount ?: (
            Deck24.cards.size - deckCount - hand.size - discarded.size - currentTable.size
        ).coerceIn(0, Deck24.cards.size)

        val impossible = hand + currentTable + discarded
        val candidates = Deck24.cards - impossible
        knownOpponent = knownOpponent.intersect(candidates)
        opponentCount = opponentCount.coerceIn(knownOpponent.size, candidates.size)

        // Never claim more exact cards than can physically be in the opponent's
        // hand. If tracking was started halfway through an animation, degrade to
        // "possible" instead of showing false certainty.
        if (knownOpponent.size > opponentCount) knownOpponent = emptySet()

        // With an empty deck, if the candidate count equals the number of hidden
        // cards, elimination has determined the complete opponent hand.
        if (deckCount == 0 && candidates.size == opponentCount) knownOpponent = candidates

        state = GameKnowledge(
            hand = hand,
            table = observation.table,
            trump = trump,
            deckCount = deckCount.coerceIn(0, 12),
            opponentCount = opponentCount.coerceAtLeast(0),
            knownOpponent = knownOpponent,
            possibleOpponent = candidates,
            discarded = discarded,
            phase = resolvePhase(observation),
            confidence = observation.confidence
        )
        return state
    }

    private fun stableSet(new: Set<Card>, old: Set<Card>, confidence: Float): Set<Card> = when {
        confidence >= 0.72f -> new
        new.isEmpty() -> old
        else -> new + old.intersect(new)
    }

    private fun resolvePhase(observation: Observation): TurnPhase {
        if (observation.phase != TurnPhase.UNKNOWN) return observation.phase
        val attacks = observation.table.size
        val defenses = observation.table.count { it.defense != null }
        return when {
            attacks == 0 -> TurnPhase.ATTACK
            attacks > defenses -> TurnPhase.DEFEND
            attacks == defenses -> TurnPhase.ATTACK
            else -> TurnPhase.WAIT
        }
    }
}
