from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

FRENCH_STOPWORDS = {
    "a","à","â","abord","afin","ah","ai","aie","aient","aies","ainsi","ait","allaient","allo","allons","après","assez",
    "attendu","au","aucun","aucune","aujourd","aujourd'hui","aupres","auquel","aura","aurai","auraient","aurais","aurait",
    "auras","aurez","auriez","aurions","aurons","auront","aussi","autre","autres","aux","auxquelles","auxquels","avaient",
    "avais","avait","avant","avec","avez","aviez","avions","avoir","avons","ayant","beaucoup","bien","bigre","boum","bravo",
    "brrr","c","ça","ca","car","ce","ceci","cela","celle","celle-ci","celle-là","celles","celles-ci","celles-là","celui",
    "celui-ci","celui-là","cent","cependant","certain","certaine","certaines","certains","certes","ces","cet","cette","ceux",
    "ceux-ci","ceux-là","chacun","chaque","cher","chère","chères","chers","chez","chiche","chut","ci","cinq","cinquantaine",
    "cinquante","cinquantième","cinquième","clac","clic","combien","comme","comment","comparable","comparables","compris",
    "concernant","contre","couic","crac","d","da","dans","de","debout","dedans","dehors","delà","depuis","dernier","derniere",
    "derrière","des","dès","desormais","desquelles","desquels","dessous","dessus","deux","deuxième","deuxièmement","devant",
    "devers","devra","différent","différente","différentes","différents","dire","divers","diverse","diverses","dix","dix-huit",
    "dix-neuf","dix-sept","dixième","doit","doivent","donc","dont","douze","douzième","dring","du","duquel","durant","e","effet",
    "eh","elle","elle-même","elles","elles-mêmes","en","encore","entre","envers","environ","es","ès","est","et","etant","étaient",
    "étais","était","étant","etc","été","etre","être","eu","euh","eux","eux-mêmes","excepté","fais","faisaient","faisant","fait",
    "feront","fi","flac","floc","font","gens","ha","hé","hein","hem","hep","hi","ho","holà","hop","hormis","hors","hou","houp",
    "hue","hui","huit","huitième","hum","hurrah","hé","i","ici","il","ils","importe","j","je","jusqu","jusque","juste","l",
    "la","là","laquelle","las","le","lequel","les","lès","lesquelles","lesquels","leur","leurs","longtemps","lors","lorsque","lui",
    "lui-même","m","ma","maint","mais","malgré","me","même","mêmes","merci","mes","mien","mienne","miennes","miens","mille","mince",
    "moi","moi-même","moins","mon","moyennant","n","na","ne","néanmoins","neuf","neuvième","ni","nombreuses","nombreux","non","nos",
    "notamment","notre","nôtre","nôtres","nous","nous-mêmes","nul","o","ô","oh","ohé","olé","ollé","on","ont","onze","onzième","ore",
    "ou","où","ouf","ouias","oust","ouste","outre","p","paf","pan","par","parce","parfois","parle","parlent","parmi","partant","pas",
    "passé","pendant","personne","peu","peut","peuvent","peux","pff","pfft","pfut","pif","plein","plouf","plus","plusieurs","plutôt",
    "pouah","pour","pourquoi","premier","première","premièrement","pres","près","proche","psitt","puis","puisque","q","qu","quand",
    "quant","quanta","quant-à-soi","quarante","quatorze","quatre","quatre-vingt","quatrième","quatrièmement","que","quel","quelle",
    "quelles","quelque","quelques","quelqu'un","quels","qui","quiconque","quinze","quoi","quoique","r","revoici","revoilà","rien",
    "s","sa","sacrebleu","sans","sapristi","sauf","se","sein","seize","selon","sept","septième","sera","serai","seraient","serais",
    "serait","seras","serez","seriez","serions","serons","seront","ses","seul","seule","seulement","si","sien","sienne","siennes",
    "siens","sinon","six","sixième","soi","soi-même","soit","soixante","son","sont","sous","stop","suis","suivant","sur","surtout",
    "t","ta","tac","tant","te","té","tel","telle","tellement","telles","tels","tenant","tes","tic","tien","tienne","tiennes","tiens",
    "toc","toi","toi-même","ton","touchant","toujours","tous","tout","toute","toutes","treize","trente","très","trois","troisième",
    "troisièmement","trop","tu","u","un","une","unes","uns","v","va","vais","vas","vé","vers","via","vif","vifs","vingt","vivat",
    "vive","vives","vlan","voici","voilà","vont","vos","votre","vôtre","vôtres","vous","vous-mêmes","vu","w","x","y","z","zut",
    "d'", "l'", "j'", "c'", "n'", "qu'", "s'", "m'", "t'"
}
ENGLISH_STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","for","on","in","to","of","by","with","without","at","from","into","out",
    "about","after","before","between","during","through","over","under","this","that","these","those","is","are","was","were",
    "be","been","being","do","does","did","doing","have","has","had","having","i","you","he","she","it","we","they","me","him",
    "her","them","my","your","his","their","our","us","am","will","would","can","could","should","as","not","no","yes","very",
    "too","so","than","such","just","also","all","any","each","few","more","most","other","some","own","same","only","again"
}
ALL_STOPWORDS = FRENCH_STOPWORDS | ENGLISH_STOPWORDS

def strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.replace("’", "'")
    text = re.sub(r"[^0-9a-zàâäçéèêëîïôöùûüÿñæœ' -]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_for_nlp(text: str) -> List[str]:
    text = clean_text(text)
    tokens = []
    for tok in text.split():
        tok = tok.strip("'-")
        if not tok:
            continue
        if tok in ALL_STOPWORDS:
            continue
        if len(tok) <= 1:
            continue
        if tok.isdigit():
            continue
        tokens.append(tok)
    return tokens

def normalize_for_match(text: str) -> str:
    return strip_accents(clean_text(text))
