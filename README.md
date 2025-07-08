# Διπλωματική Εργασία  
## Μεταπτυχιακό Πρόγραμμα: Data Science and Business Analytics  
### Πανεπιστήμιο Μακεδονίας

**Τίτλος διπλωματικής**: Development and Optimization of a Neural Network for the Detection of Verticillium Disease in Olive Trees

Το παρόν αποθετήριο περιλαμβάνει τα Jupyter Notebooks που αναπτύχθηκαν στο πλαίσιο της διπλωματικής εργασίας του μεταπτυχιακού προγράμματος **Data Science and Business Analytics** του Πανεπιστημίου Μακεδονίας. Η εργασία επικεντρώνεται στην κατασκευή και αξιολόγηση μοντέλων βαθιάς μάθησης για τη διάγνωση της ασθένειας Verticillium σε ελαιόδεντρα από εικόνες drone.

---

## Δομή του αποθετηρίου

### 1. Τελικά Notebooks (κύρια υλοποίηση)

- [`Remastered-Optuna300-BestAcc.ipynb`](./Remastered-Optuna300-BestAcc.ipynb)  
  Τελικό μοντέλο, βελτιστοποιημένο με χρήση Optuna.

- [`Remastered Pretrained Models_LAST.ipynb`](./Remastered%20Pretrained%20Models_LAST.ipynb)  
  Σύγκριση προεκπαιδευμένων μοντέλων (π.χ. MobileNetV2, EfficientNetB0, ResNet50V2).

- [`Ablation_last.ipynb`](./Ablation_last.ipynb)  
  Ablation study για κατανόηση συνεισφοράς κάθε αρχιτεκτονικού στοιχείου.

### 2. Δευτερεύοντα Notebooks (δοκιμές / μη τελικές προσπάθειες)

📁 [`ExtraNotebooks/`](./ExtraNotebooks)

Περιλαμβάνει notebooks από ενδιάμεσες, δοκιμαστικές ή αποτυχημένες υλοποιήσεις. Διατηρούνται για λόγους καταγραφής, σύγκρισης και πιθανής μελλοντικής αξιοποίησης.

---

Όλα τα notebooks έχουν υλοποιηθεί σε Python, με χρήση βιβλιοθηκών όπως TensorFlow/Keras, Optuna, Scikit-learn και Matplotlib.
