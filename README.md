# NLP-Tool: Textklassifizierung und Zusammenfassung
## Organisatorisches

### Projektrealisierung_Gruppe5
Teammitglieder: Amina Uicker-Darwish, Aymane Bouguer, Jan Rüdt, Lukas Bonn <br>
Kurs: WWI20DSA <br>
Dozierende: Enzo Hilzinger, Michael Lang

### Projektziele: 
- Entwicklung eines NLP-Tools zur Textzusammenfassung und –klassifikation
- Zusammenfassung eines gegebenen Textes unter Angabe einer gewünschten Kompressionsrate zwischen 20% und 80%
- Klassifizierung der Texte in die Oberkategorien Nachrichtenartikel, Rechtstexte, Reden von Präsidenten, literarische Texte und Blogs
- Integration einer barrierefreien Webapp, in der Texte eingegeben werden, Dateien im TXT-, DOC(X)- und ODT-Format hochgeladen oder Texte per Spracheingabe eingefügt werden können.
  
### Kriterien:

- Verwendung eines mehrstufigen Ansatzes (z.B. Data Selection, Preprocessing und Training des
Modells) inkl. Auswahl und Anpassung eines geeigneten Algorithmus
- Evaluation des erstellten Tools anhand geeigneter Metriken (nicht nur Accuracy basierend auf
Testdaten!)
- Selbständige Auswahl einer geeigneten Toolchain, keine Verwendung von vordefinierten Tools,
die o.g. Anforderungen bereits komplett abdecken!
- Entwicklungsprozess muss nachvollziehbar gestaltet sein (via Commits auf GitHub)


## Installation
1. Verwenden des folgenden Befehl, um das Repository in ein lokales Verzeichnis zu klonen: `git clone bnnlukas/Projektrealisierung_Gruppe5`
2. Sicherstellen, dass die erforderlichen Abhängigkeiten installiert sind, indem der folgende Befehl ausgeführt wird: `pip install -r requirements.txt`
3. Zusätzliche Dokumente aus dem folgenden OneDrive-Verzeichnis herunterladen und im lokalen Projektverzeichnis eingefügen:
https://caponovawarade-my.sharepoint.com/:f:/g/personal/ruedtja_caponova_wara_de/EjJkGARKPY1Jifgyz47zhDcBhckA8x31IoJbr9b7uDt6nw?e=QOGNX9
  - finalized_model.sav in dem Order "models" einfügen
  - blogtext.csv in dem Order "data\Blogs" einfügen


## Verwendung
1. Starten der Webapp: `flask run` im lokalen Projektverzeichnis


## Beispiele























