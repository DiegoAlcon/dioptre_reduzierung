Wofür diese Datei gedacht ist: Dies ist ein reines Textdokument, das vor allem als Brainstorming für das Projekt dienen soll, das am 19.03.2024 um 13.30 Uhr in R&D Alcon besprochen wird.
 
Hintergrund: Uns liegt eine Reihe von Bildern vor, die gescannte Waffer darstellen. Jedes dieser Bilder hat einen Dioptrienfaktor, der als der Kehrwert der Entfernung definiert ist, die die Linse erreichen kann.
            Mit Hilfe von Bildverarbeitungstechniken wurde eine Reihe von parasitären Kreisen identifiziert und aus den Bildern entfernt. Dadurch wurde der Dioptrien-Faktor um einen unbekannten Betrag reduziert. 

Umfang: Bei der Füllung von Pariser Linsen entstehen Luftblasen unterschiedlicher Größe. Diese können nach dem Füllen mit Hilfe eines Vakuumofens entfernt werden.
       Dadurch verringert sich jedoch das Volumen der Linse und damit auch die Dioptrienzahl. 
       Um dem entgegenzuwirken, wird versucht, die Linse um das Volumen der Luftblase zu überfüllen.
       Es wurden erste Versuche unternommen, den Durchmesser der Luftblasen manuell mit einem Mikroskop zu messen und das Volumen der Luftblase zu bestimmen.
       Das Ergebnis war jedoch nicht aussagekräftig. 
       Welches Ergebnis? Das Ergebnis bezüglich der Genauigkeit der Ergebnisse oder die Menge des Volumens, die überfüllt werden muss?
       Es stellt sich die Frage, ob es sich wirklich um eine sphärische Erscheinung handelt und ob das Aussehen der gesamten Linse einen Einfluss darauf hat.
       --> Ziel dieser Arbeit ist es, modernere Methoden der Bildverarbeitung auszuwählen und zu untersuchen, um herauszufinden, ob es eine Korrelation (zwischen dem Durchmesser der Blasen und der reduzierten Dioptrie) gibt oder
           ob eine Vorhersage der Dioptrienveränderung des Blasenentfernungsprozesses möglich ist. 

Hauptaufgabe: Konstruieren Sie einen einfachen Algorithmus, der jedes beliebige Bild dieser Art (vor und nach der Entfernung des parasitären Kreises) mit dem zugehörigen Dioptrienfaktor (Etiketten) identifiziert 
           der jedes beliebige Bild dieser Art (vor der Entfernung der Blase) identifiziert und die Dioptrienreduktion vorhersagt.

Verwendung: 
    Eingabe: Bild mit "Blase" (auch bekannt als Blasen oder Kreise) 
    Ausgabe: Reduktion des Dioptrienfaktors.

Training: 
    Eingabe: Bilder mit und ohne "Blase" mit ihren jeweiligen Beschriftungen. 

Berücksichtigen Sie die folgenden Schritte: 

Vorverarbeiten der Bilder:
    Lesen Sie die Bilder.
    Wenden Sie alle erforderlichen Bildverarbeitungstechniken an (z. B. Rauschunterdrückung, Kantenerkennung), um die Merkmale zu verbessern.
    Falls noch nicht geschehen, identifizieren und entfernen Sie parasitäre Kreise aus den Bildern.
Merkmalsextraktion:
    Extrahieren relevanter Merkmale aus den vorverarbeiteten Bildern. Diese Merkmale werden für die Vorhersage der Dioptrienfaktoren verwendet.
    Mögliche Merkmale sind:
        Intensitätsverteilung (Histogramm): Analysieren Sie die Pixelintensitätswerte im gesamten Bild.
        Geometrische Merkmale: Extrahieren von Eigenschaften der identifizierten Kreise (z. B. Fläche, Umfang, Kreisform).
        Texturmerkmale (z. B. Haralick-Merkmale): Beschreiben die Texturmuster innerhalb des Bildes.
        Andere bereichsspezifische Merkmale: Berücksichtigen Sie alle zusätzlichen Merkmale, die für Ihren spezifischen Datensatz relevant sind.

        Aufteilung des Datensatzes:
        Aufteilung des Datensatzes in Trainings- und Testteilmengen.
    Auswahl des Modells für maschinelles Lernen und Training:
        Wählen Sie einen geeigneten Algorithmus für maschinelles Lernen (für Regression, da es sich um ein Problem mit kontinuierlicher Ausgabe handelt).
        Trainieren Sie das Modell anhand der Trainingsdaten und der extrahierten Merkmale.
        Passen Sie die Hyperparameter an, falls erforderlich.
    Bewertung des Modells:
        Evaluieren Sie die Leistung des Modells anhand der Testdaten.
        Verwenden Sie geeignete Bewertungsmetriken (z. B. mittlerer quadratischer Fehler für Regression, Genauigkeit für Klassifikation).
    Vorhersage von Dioptrien-Faktoren für neue Bilder:
        Gegeben ein neues Bild:
            Vorverarbeitung des Bildes (ähnlich wie bei den Trainingsbildern).
            Extrahieren relevanter Merkmale.
            Verwenden Sie das trainierte Modell zur Vorhersage des Dioptrienfaktors.
    
    Fragen, die Sie sich stellen sollten:
        Welche Programmiersprache soll verwendet werden? --> Python
        In welchem Format sollen die Bilder geliefert werden? --> .jpg
        Soll dies zu einer Online-Anwendung weiterentwickelt werden?
        Wenn es für die Online-Nutzung vorgesehen ist, welche zeitaufwändigen Kriterien sollten berücksichtigt werden?
        Eignet sich dieses Problem gut für Techniken des maschinellen Lernens?
        Wenn ja, welche Bibliotheken sollten verwendet werden? Welche Programmierungsebene ist erforderlich?
        Welche Art von Algorithmus für maschinelles Lernen sollte verwendet werden? Regression? Klassifizierung? 

        Überlegen Sie sich eine Hypothese darüber, was den Dioptrienfaktor jedes Bildes beeinflussen könnte (wichtig für die Merkmalsextraktion)
        Das Vorhandensein dieses parasitären Kreises: [1= Vorhandensein, 0= Abwesenheit]
        Die Größe des parasitären Kreises: Radien
        Die Form des Störkreises: wir nehmen an, dass alle Kreise sind
        Die Anzahl dieser parasitären Kreise: 
        Die durchschnittliche Pixelintensität dieses parasitären Kreises. 
    
    Überlegen Sie sich einen verallgemeinerten Kaskode für eine mögliche Implementierung.
        Image Pulling:
            Konstruieren Sie eine Funktion, die die Bilder aus einem Ordner liest.
            Vor der Konstruktion der Hauptarrays müssen die Dimensionen aller Bilder abgeglichen werden.
                Dazu wird eine große Anzahl von Waffeldurchmessern (d) gemittelt, um Bilder der Größe dxd zu konstruieren, die im Waffelzentrum zentriert sind.
            Konstruieren Sie zwei Arrays der Form: 
                with_bubble = x_dim x y_dim x n_images oder x_dim x y_dim x z_dim (3) x n_dim (falls nicht bereits in Graustufen)
                without_bubble = x_dim x y_dim x m_images oder x_dim x y_dim x z_dim (3) x n_dim (falls nicht bereits in Graustufen)
        Vorverarbeitung der Bilder:
            Konvertierung in Graustufen.
            Artefaktunterdrückung durchführen:
                Bilder können auch verrauscht sein. Bei verrauschten Bildern muss entweder das Rauschen beseitigt oder die Probe gelöscht werden.
                Einige Ablehnungskriterien sind (speziell für dieses Beispiel):
                Fehlen einer gut definierten Waffel in der Mitte des Bildes.
                Abweichung von der durchschnittlichen Gesamtenergie über einem bestimmten Schwellenwert.
            Um diese Unannehmlichkeiten zu beheben, wenden Sie einen Butterworth-Hochpassfilter an und bewerten die Verbesserungen.
        Mischen Sie beide Arrays auf:
            images = x_dim x y_dim x (n+m)m_images
            training_images = 0.8*(Bilder)
            validation_images = 0.2*(Bilder)
        Konstruieren Sie eine Etikettenliste wie folgt:
            labels = [diopter1, diopter2, ..., diopter(n+m)]
            training_labels = 0.8*(labels)
            validation_labels = 0.2*(labels)
        Um eine Klassifizierungsaufgabe zu implementieren, muss die Auflösung der Ausgabe groß sein. 
        Bildtraining:
        Entscheiden Sie, welche Programmiersprache Sie verwenden wollen: Python in VS, verbunden mit einem Git-Respository.
        Entscheiden Sie, welchen Algorithmus Sie verwenden wollen: 
            Visual Geometry Group (Convolutional Neural Network) (Klassifizierungsbezogen)
                Diese Implementierung ist zeitaufwändig, für Online-Anwendungen sollte eine schnellere Architektur in Betracht gezogen werden. Diese Implementierung ist nicht gut geeignet, da wir es mit einem Regressionsproblem zu tun haben.
                Aber vielleicht ist es wertvoll, sie zu implementieren, da eine ausreichend gute quasi-kontinuierliche Auflösung erreicht werden kann, indem der Ausgaberaum in ausreichend kleine Intervalle unterteilt wird, von denen jedes eine Klassifikation darstellt.
            Neuronales Regressionsnetzwerk gekoppelt mit einer Hough-Transform-Merkmalsextraktion (HTFT)
                (Diese Implementierung ist gut für Regressionsprobleme geeignet)
                HTFT:
                    Eine Reihe von Merkmalen, die mit Hilfe der von der Hough-Transformation gelieferten Informationen extrahiert werden können, wie z. B.:
                        - Vorhandensein von Kreisen.
                        - Position des Kreises (x, y) kooriniert.
                        - Radien der Kreise.
                        - Anzahl der Kreise.
                        - Durchschnittliche Pixelintensität innerhalb des Kreises.
                        Alle diese Merkmale sollten in einem Merkmalsvektor gruppiert und auf den Bereich [0 1] normalisiert werden, ohne die ursprüngliche relative Abweichung zu beeinträchtigen. 
                        Die innere Eingabe des Modells (nach Anwendung der Hough-Transformation auf das Originalbild) ist dieser Merkmalsvektor.
                        Konstruieren Sie eine Reihe von vollständig verknüpften Schichten und passen Sie die Gewichte jedes Merkmals an, indem Sie Backpropagation mit einer Fehlerfunktion auf der Grundlage der kleinsten mittleren Quadrate anwenden, um gute Regressionsergebnisse zu erzielen.
                        Prüfen Sie verschiedene Modellimplementierungen (linear, quadratisch, etc.)
                        Erarbeitung eines mathematischen Modells, um jedes Eingangsbild mit der Änderung des Dioptrienfaktors zu korrelieren.
            Deklarieren Sie die Hyperparameter des Netzes: 