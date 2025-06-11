
def check_duplicates(liste):
    return len(liste) != len(set(liste))

def trouver_articles_manquants(articles_extraits, articles_theoriques):
    extraits_set = set(a.strip() for a in articles_extraits)
    existants_set = set(a.strip() for a in articles_theoriques)

    manquants = sorted(existants_set - extraits_set)
    return manquants

def trouver_articles_extras(articles_extraits, articles_theoriques):

    return sorted(set(articles_extraits) - set(articles_theoriques))