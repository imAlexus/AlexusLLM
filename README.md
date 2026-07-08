# AlexusLLM

LLM costruito **da zero** (transformer decoder-only in PyTorch) — capace di
conversare e, con dati di ragionamento, di seguire catene di pensiero.
Progettato per essere allenato e sperimentato su **Google Colab**.

Autore: imAlexus

## Cosa c'e dentro

- `model.py` — Transformer decoder-only from scratch (attention, RMSNorm,
  weight tying, campionamento con temperatura + top-k).
- `tokenizer.py` — Tokenizer **BPE** scritto da zero, addestrato sui tuoi dati
  (IT+EN). Niente dipendenze esterne pesanti.
- `data.py` — helper per caricare il corpus conversazionale.
- `train.py` — loop di training minimale (CPU in minuti, GPU Colab in secondi).
- `chat.py` — chat interattiva sul checkpoint allenato.
- `gen_sample.py` — genera un piccolo dataset IT+EN di esempio (conversazione).
- `gen_reasoning.py` — genera esempi di **chain-of-thought** (ragionamento a
  passi) da appendere al corpus per insegnare al modello a ragionare.
- `colab/AlexusLLM_Train.ipynb` — notebook pronto da aprire in Colab.

## Avvio rapido (locale)

```bash
pip install -r requirements.txt
python gen_sample.py          # crea data/sample.txt
python train.py --data data/sample.txt --epochs 3
python chat.py
```

## Su Google Colab

Apri `colab/AlexusLLM_Train.ipynb` (clone automatico da questo repo) oppure
incolla nel notebook:

```python
!git clone https://github.com/imAlexus/AlexusLLM
%cd AlexusLLM
!pip install -r requirements.txt
!python gen_sample.py
!python train.py --data data/sample.txt --epochs 5 --device cuda
```

> Per usare la GPU gratuita di Colab: Runtime -> Change runtime type -> GPU.

## Dataset: conversazione + ragionamento

```bash
python gen_sample.py              # conversazione IT+EN
python gen_reasoning.py >> data/sample.txt   # aggiunge esempi di ragionamento
python train.py --bpe --data data/sample.txt --epochs 5 --dim 384 --n_layers 8
```

Il formato e sempre:

    USER: <testo>
    ASSISTANT: <risposta, eventualmente con Passo 1 / Passo 2 ...>

Il modello impara cosi a scomporre i problemi (reasoning) e non solo a
completare pattern.

Parametri in `train.py`: `--dim`, `--n_layers`, `--n_heads`, `--vocab_size`,
`--block_size`, `--batch_size`. Un modello davvero capace richiede dataset
molto piu grandi (milioni di token) e piu parametri — questo repo e il punto
di partenza didattico.

## Roadmap ragionamento

Per insegnare a "ragionare" (non solo completare):
1. Aggiungi esempi con `USER: <problema>` / `ASSISTANT: <passo 1> ... <risposta>`
   (chain-of-thought).
2. Aumenta `--block_size` per sequenze piu lunghe.
3. (Avanzato) RL tipo reward model sui passaggi corretti.

## Licenza

MIT — sperimenta liberamente.
