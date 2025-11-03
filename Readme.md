
NAME : ```PAC-srsk-1729```

```export PYTHONPATH=$PYTHONPATH:.```

```export METAMON_CACHE_DIR="PAC-dataset"```

```modal shell --volume my-volume```


👁️U👁️

model_ckpts: https://huggingface.co/jakegrigsby/metamon/tree/main

modal volume get pokemon-showdown-gen1 results/HRM_Pokemon_Gen1/ckpts/latest/policy.pt  model_weights.pt

---

### Detailed Comparison of Pokémon Battle Mechanics: Gen 1 vs. Gen 9

| Feature                  | Generation 1 (RBY) Detailed Explanation                                                                                                                                                                                                                                                              | Generation 9 (Scarlet & Violet) Detailed Explanation                                                                                                                                                                                                                                                                                                                                                           |
| :----------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Stats** 📊            | **Unified "Special" Stat**: A single stat governed both special attacks and defense. A Pokémon with a high Special, like `Alakazam`, was automatically both a powerful special attacker and a resilient special defender. This limited role diversity, as there was no such thing as a "special wall" that wasn't also a strong special attacker. | **Split Special Stats**: The "Special" stat was split into **Special Attack (SpA)** and **Special Defense (SpD)**. This created far more specialized roles. A Pokémon like `Blissey` can have titanic SpD to absorb special hits but mediocre SpA, making it a pure special wall. Conversely, a Pokémon like `Chi-Yu` has devastating SpA but is defensively frail. This split is fundamental to modern team building. |
| **Abilities** ✨          | **Did not exist**. A Pokémon was defined entirely by its stats, typing, and movepool. A `Charizard` was identical to every other `Charizard` in this regard.                                                                                                                                      | **Inherent Passive Skills**: Every Pokémon has at least one Ability that provides a passive effect in battle. These are non-negotiable and define a Pokémon's role. For example, `Dragonite's` **Multiscale** halves damage taken at full HP, making it an excellent setup sweeper. `Great Tusk's` **Protosynthesis** boosts its best stat in harsh sunlight. Abilities add a massive layer of strategy, creating synergies and counters. |
| **Held Items** 🎒        | **Did not exist**. Pokémon fought empty-handed. The battle was a direct test of one Pokémon's stats and moves against another's.                                                                                                                                                                 | **Equippable Battle Enhancers**: Pokémon can hold one of hundreds of items that provide a huge range of effects. This adds immense customization. For instance: **Choice Band** boosts a Pokémon's Attack by 50% but locks it into the first move it uses. **Leftovers** provides passive healing, restoring 1/16th of max HP each turn. **Booster Energy** is a one-time use item that activates a Paradox Pokémon's ability. Items can completely change how a Pokémon functions. |
| **Natures** 🍃            | **Did not exist**. A Pokémon's stats were calculated directly from its base stats, level, IVs, and EVs. There was no personality-based modification.                                                                                                                                                       | **Stat-Altering Personalities**: Every Pokémon has a Nature that boosts one stat by 10% and lowers another by 10% (some are neutral). This is crucial for optimization. A physical attacker like `Scizor` will almost always have an **Adamant Nature** (+Attack, -SpA) to maximize its damage output. A fast, frail attacker like `Deoxys-Speed` will use a **Timid Nature** (+Speed, -Attack) to ensure it moves first. |
| **EVs & IVs** 🧬         | **Primitive and Opaque**: **"Stat Experience"** (the precursor to EVs) was gained in every stat for every Pokémon defeated, making it impossible to target-train a single stat. **"DVs"** (precursor to IVs) were a hidden value from 0-15 for each stat that determined its quality. They also determined a Pokémon's Hidden Power type, but were not manipulable. | **Transparent and Precise**: **Effort Values (EVs)** are a total of 510 points you can distribute to a Pokémon's stats (max 252 in one stat) by defeating specific Pokémon or using vitamins. This allows for precise customization of a Pokémon's stat spread. **Individual Values (IVs)** are a "gene" value from 0-31 for each stat. In modern games, they are visible and can even be maximized via "Hyper Training." |
| **Generation Gimmick** 💎 | **None**. The battle system was consistent and straightforward, with no temporary, battle-altering power-ups.                                                                                                                                                                                             | **Terastal Phenomenon**: Once per battle, a Pokémon can **Terastallize**, changing its type to its designated "Tera Type." This has profound strategic implications. **Defensive Use:** A Dragon-type Pokémon like `Dragonite` can become **Tera Steel** to resist the Ice, Dragon, and Fairy moves it's normally weak to. **Offensive Use:** A Pokémon can become a Tera Type that matches one of its attacks to gain a huge power boost (STAB - Same Type Attack Bonus), like `Great Tusk` using **Tera Ground** to make its Headlong Rush overwhelmingly powerful. |
| **Move Mechanics** ⚙️     | **Quirky and Often Broken**: Many moves had unique and exploitable mechanics. **Hyper Beam**: This powerful move required no recharge turn if it knocked out the opponent, making it an incredible finishing move. **Wrap/Bind**: These moves didn't just do damage; they prevented the opponent from doing *anything* for 2-5 turns. **Freeze**: This status was essentially permanent unless the frozen Pokémon was hit by a Fire-type move. | **Balanced and Standardized**: Move mechanics have been refined for better competitive balance. **Hyper Beam** now *always* requires a recharge turn, making it a high-risk move. **Trapping Moves** like Whirlpool or Fire Spin now only prevent switching; the trapped Pokémon can still attack. **Freeze** now has a ~20% chance to thaw out on its own each turn. These changes promote more interactive and less frustrating gameplay. |

---

### **Game State Keywords**

* **no-effect**: This indicates that no specific effect is currently active on the Pokémon, such as a stat boost or debuff from a move like "Swords Dance" or "Screech." If a Pokémon used a move that raised its attack, this would likely change.

* **no-status**: This means the active Pokémon does not have a major status condition. Major status conditions in Pokémon are debilitating ailments that last for the duration of the battle (unless cured), such as:
    * **Burn** (🔥): Reduces a Pokémon's physical attack power and damages it each turn.
    * **Freeze** (🧊): Prevents a Pokémon from moving.
    * **Paralyze** (⚡): Halves a Pokémon's speed and gives it a chance to be unable to move.
    * **Poison** (☠️): Damages a Pokémon each turn.
    * **Sleep** (😴): Prevents a Pokémon from moving for a number of turns.

* **no-weather**: This signifies that there is no weather condition currently affecting the battlefield. Weather in Pokémon can be created by certain moves or abilities and has various effects on different types of Pokémon and moves. Examples include:
    * **Rain Dance** (🌧️): Boosts Water-type moves and weakens Fire-type moves.
    * **Sunny Day** (☀️): Boosts Fire-type moves and weakens Water-type moves.
    * **Sandstorm** (🌪️): Damages non-Ground, Rock, or Steel-type Pokémon each turn.
    * **Hail** (❄️): Damages non-Ice-type Pokémon each turn.

* **no-conditions**: This is a more general term that encompasses no specific field conditions. While "no-weather" refers to a specific type of field condition, "no-conditions" likely includes the absence of other temporary effects on the field, such as **Stealth Rock** (a move that damages an opponent's Pokémon as it switches in) or **Spikes** (a similar move that inflicts damage on grounded Pokémon as they switch in).

***

### **The Long String of Words**

The long string of words after the player and opponent tags is a verbose way of describing the **current state of the game**, but it is in a format that's easy for a computer to process. This "text observation" is a snapshot of all the relevant information at that moment in the battle.

Let's break down the string from the point of view of **King Wynaut**:

* **&lt;player&gt; piloswine lifeorb oblivious ground ice noeffect nostatus**: This describes King Wynaut's active Pokémon.
    * **piloswine**: The name of the Pokémon.
    * **lifeorb**: An item Piloswine is holding that boosts its attack at the cost of some HP.
    * **oblivious**: The Pokémon's ability.
    * **ground ice**: The two Pokémon types (or "typings") of Piloswine.
    * **noeffect nostatus**: As explained above, no special effects or status conditions are active.

* **&lt;move&gt; avalanche ice physical**: This is a move available to the Pokémon.
    * **avalanche**: The name of the move.
    * **ice**: The type of the move.
    * **physical**: The category of the move (Physical vs. Special).

The string continues to list all available moves for the active Pokémon (**Earthquake**, **Stealth Rock**, **Stone Edge**), then describes the rest of King Wynaut's team on the bench, one Pokémon at a time, followed by the opponent's team and their moves. For example, **&lt;switch&gt; haunter lifeorb levitate &lt;moveset&gt; shadowball sludgebomb substitute thunderbolt** shows that King Wynaut has a Haunter in his team, holding a Life Orb, with the ability Levitate, and the moves Shadow Ball, Sludge Bomb, Substitute, and Thunderbolt.

---

The strings starting with "no-..." (nostatus, noeffect, noitem, noability, notype) are part of a standardized, universal vocabulary used by the Metamon framework.

Purpose: They act as canonical placeholders for the absence of a particular game condition. For a machine learning model, which requires fixed-size inputs, it's crucial to have a consistent way to represent "nothing." Instead of using None or an empty string, which can be inconsistent or cause errors, these strings provide a uniform token that the model can easily recognize and process.

Example: If a Pokémon is not burned or poisoned, its status attribute will be "nostatus". If a Pokémon is holding no item, its item attribute will be "noitem". This provides a predictable, non-empty value for every attribute, simplifying the process of converting the dataclass into a feature vector for an AI.
---