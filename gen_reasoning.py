"""gen_reasoning.py — adds chain-of-thought (reasoning) examples to the corpus.

These show the model how to break a problem into steps before answering, which
is the core of "reasoning" behavior (mirrors how models like DeepSeek-R1 / o1
are trained). Format stays USER/ASSISTANT so train.py consumes it directly.

Run:
    python gen_reasoning.py >> data/sample.txt
or generate a separate file and pass both to train.py.
"""
import random

REASONING = [
    ("Quanto fa (14 + 6) * 3 ?",
     "Passo 1: calcolo dentro le parentesi, 14 + 6 = 20. "
     "Passo 2: moltiplico il risultato per 3, 20 * 3 = 60. "
     "Risposta: 60."),
    ("Se ho 5 mele, ne compro 3 e ne do 2 a un amico, quante ne ho?",
     "Passo 1: parto da 5 mele. "
     "Passo 2: ne compro 3, quindi 5 + 3 = 8. "
     "Passo 3: ne do 2, quindi 8 - 2 = 6. "
     "Risposta: 6 mele."),
    ("Un treno va a 60 km/h per 2 ore e mezza. Quanta strada fa?",
     "Passo 1: converto il tempo in ore, 2 ore e mezza = 2.5 h. "
     "Passo 2: moltiplico velocita per tempo, 60 * 2.5 = 150. "
     "Risposta: 150 km."),
    ("Se oggi e martedi, che giorno sara tra 10 giorni?",
     "Passo 1: una settimana ha 7 giorni, quindi 10 giorni = 1 settimana + 3 giorni. "
     "Passo 2: martedi + 3 giorni = venerdi. "
     "Risposta: venerdi."),
    ("Qual e il quadrato di 13?",
     "Passo 1: 13 al quadrato significa 13 * 13. "
     "Passo 2: 13 * 10 = 130, 13 * 3 = 39, sommo 130 + 39 = 169. "
     "Risposta: 169."),
    ("In una scatola ci sono 3 file da 4 palline ciascuna. Quante palline in totale?",
     "Passo 1: ho 3 file. "
     "Passo 2: ogni fila ha 4 palline. "
     "Passo 3: moltiplico 3 * 4 = 12. "
     "Risposta: 12 palline."),
    ("If a shirt costs 20 and is 25% off, what is the price?",
     "Step 1: 25% of 20 is 20 * 0.25 = 5. "
     "Step 2: subtract the discount, 20 - 5 = 15. "
     "Answer: 15."),
    ("A rectangle is 5 wide and 8 tall. What is its area?",
     "Step 1: area of a rectangle is width times height. "
     "Step 2: 5 * 8 = 40. "
     "Answer: 40."),
    ("How many minutes are there in 3 hours and 20 minutes?",
     "Step 1: 1 hour is 60 minutes, so 3 hours is 3 * 60 = 180 minutes. "
     "Step 2: add the extra 20 minutes, 180 + 20 = 200. "
     "Answer: 200 minutes."),
    ("What is 100 divided by 4, then add 7?",
     "Step 1: 100 / 4 = 25. "
     "Step 2: add 7, 25 + 7 = 32. "
     "Answer: 32."),
]


def main():
    random.seed(7)
    lines = []
    for _ in range(15):  # repeat for more weight on reasoning
        for u, a in REASONING:
            lines.append(f"USER: {u}\nASSISTANT: {a}\n")
    random.shuffle(lines)
    print("".join(lines), end="")


if __name__ == "__main__":
    main()
