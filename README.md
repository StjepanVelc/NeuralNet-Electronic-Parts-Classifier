# NeuralNet Electronic Parts Classifier

CNN klasifikator elektroničkih komponenti s transfer learning pristupom.

## Brzi start

1. Instalacija paketa:

```bash
pip install -r requirements.txt
```

2. Trening modela:

```bash
python -m src.train --model resnet18
```

3. Evaluacija na test skupu:

```bash
python -m src.evaluate
```

4. Predikcija za jednu sliku:

```bash
python -m src.predict path/to/image.jpg --top-k 3
```

5. Pokretanje Streamlit aplikacije:

```bash
streamlit run app/streamlit_app.py
```

## Što skripta radi automatski

- Ako split ne postoji, automatski radi split iz `data/raw` u `data/train`, `data/val`, `data/test`.
- Koristi augmentacije i weighted sampling zbog neuravnoteženih klasa.
- Sprema najbolji model u `models/best_model.pt`.
- Sprema metrike u `reports/metrics.json`.
- Evaluacija sprema confusion matrix u `reports/figures/confusion_matrix.png`.

