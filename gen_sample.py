"""gen_sample.py — generate a small IT+EN conversational + reasoning dataset.

Produces data/sample.txt in the format the trainer expects:

    USER: ...
    ASSISTANT: ...

The data is synthetic (templates) so the repo runs out-of-the-box. For a real
model you replace this with a scraped/curated corpus.
"""
import random

TEMPLATES_IT = [
    ("Ciao, come stai?", "Ciao! Sto bene, grazie. Sono AlexusLLM, un modello allenato da zero. Dimmi pure come posso aiutarti."),
    ("Spiegami cosa e una rete neurale.", "Una rete neurale e un modello composto da strati di unita chiamate neuroni. Ogni neurone combina gli input con dei pesi, applica una funzione di attivazione e passa il risultato al livello successivo. Imparando dai dati, la rete adjusta i pesi per ridurre lerrore sulle previsioni."),
    ("Quanto fa 17 piu 25?", "17 piu 25 fa 42. Sommiamo le unita: 7+5=12, poi 10+20=30, e 30+12=42."),
    ("Perche il cielo e blu?", "Il cielo appare blu perche la luce del sole viene dispersa dallatmosfera. Le molecole daria disperdono piu efficacemente le lunghezze donda corte, cioe il blu, rispetto a quelle lunghe come il rosso."),
    ("Come posso imparare a programmare?", "Inizia con le basi: variabili, cicli e funzioni. Poi costruisci piccoli progetti. La pratica costante batte qualsiasi corso teorico. Vuoi che ti mostri un esempio di codice?"),
    ("Cos e lattention nei transformer?", "Lattention permette al modello di pesare limportanza di ogni parola rispetto alle altre nella stessa frase. Invece di trattare le parole in isolamento, ogni token guarda tutto il contesto e decide a cosa dare piu attenzione."),
    ("Risolvi: se ho 3 mele e ne mangio 1, quante ne restano?", "Ne restano 2. Parti da 3, sottrai 1 mangiata, e ottieni 3-1=2."),
    ("Cosa significa allenare un modello?", "Allenare un modello significa mostrargli molti esempi e fargli regolare i propri parametri per prevedere bene i dati. Si misura lerrore e lo si riduce aggiornando i pesi tramite discesa del gradiente."),
    ("Dimmi una curiosita.", "Lo sapevi che il primo programma informatico della storia e stato scritto da Ada Lovelace nel 1800? Aveva previsto che le macchine avrebbero potuto comporre musica e creare arte."),
    ("Come funziona la memoria di un computer?", "La memoria conserva dati come sequenze di bit, cioe zeri e uni. La RAM e veloce ma volatile, mentre il disco e lento ma persistente. Il processore legge e scrive in memoria per eseguire i programmi."),
]

TEMPLATES_EN = [
    ("Hi, what is your name?", "Hi! I am AlexusLLM, a language model trained from scratch. How can I help you today?"),
    ("Explain what a neural network is.", "A neural network is a model made of layers of units called neurons. Each neuron combines inputs with weights, applies an activation function, and passes the result forward. By learning from data, the network tunes its weights to reduce prediction error."),
    ("What is 12 times 8?", "12 times 8 is 96. You can think of it as 10 times 8 equals 80, plus 2 times 8 equals 16, and 80 plus 16 is 96."),
    ("Why is the sky blue?", "The sky looks blue because air molecules scatter sunlight. Shorter wavelengths like blue scatter more strongly than longer ones like red, so blue light fills the sky."),
    ("How do transformers work?", "A transformer uses self-attention so every token can look at every other token in the sequence and decide what to focus on. Stacks of attention and feed-forward layers build up understanding of the text."),
    ("What does it mean to train a model?", "Training means showing a model many examples and adjusting its parameters so it predicts well. You measure the error and reduce it by updating weights through gradient descent."),
    ("Tell me a fun fact.", "Did you know the first computer program was written by Ada Lovelace in the 1800s? She imagined machines could one day compose music and make art."),
    ("What is reasoning in AI?", "Reasoning is the ability to combine facts step by step to reach a conclusion, rather than just recalling patterns. Models can be trained on step-by-step examples to imitate this chain-of-thought behavior."),
    ("How can I learn Python?", "Start with variables, loops, and functions, then build small projects. Consistent practice beats any theory-only course. Want me to show you a short example?"),
    ("What is attention?", "Attention lets a model weight how relevant each word is to the others in a sentence. Instead of reading words in isolation, every token attends to the full context and decides what matters most."),
]


def main():
    random.seed(42)
    lines = []
    # repeat and lightly shuffle to grow the corpus
    for _ in range(25):
        for u, a in TEMPLATES_IT + TEMPLATES_EN:
            lines.append(f"USER: {u}\nASSISTANT: {a}\n")
    # shuffle conversation order for variety
    random.shuffle(lines)
    out = "".join(lines)
    with open("data/sample.txt", "w", encoding="utf-8") as f:
        f.write(out)
    print(f"wrote data/sample.txt ({len(out):,} chars, {len(lines)} turns)")


if __name__ == "__main__":
    main()
