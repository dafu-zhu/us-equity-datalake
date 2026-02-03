"""
Loughran-McDonald Financial Sentiment Word Lists.

Based on the Master Dictionary from https://sraf.nd.edu/loughranmcdonald-master-dictionary/
These word lists are specifically designed for financial text analysis.

Reference:
Loughran, T., & McDonald, B. (2011). When is a liability not a liability?
Textual analysis, dictionaries, and 10-Ks. Journal of Finance, 66(1), 35-65.
"""

# Uncertainty words - indicate vagueness or imprecision
# Associated with higher stock return volatility
UNCERTAINTY_WORDS = {
    "almost", "ambiguity", "ambiguous", "anticipate", "anticipated",
    "apparent", "apparently", "appear", "appeared", "appearing", "appears",
    "approximate", "approximated", "approximately", "approximates",
    "approximating", "arbitrarily", "arbitrary", "assume", "assumed",
    "assumes", "assuming", "assumption", "assumptions", "believe",
    "believed", "believes", "believing", "cautious", "cautiously",
    "conceivable", "conceivably", "conditional", "conditionally",
    "confuse", "confused", "confuses", "confusing", "confusion",
    "contingencies", "contingency", "contingent", "contingently",
    "could", "depend", "depended", "dependence", "dependencies",
    "dependency", "dependent", "depending", "depends", "destabilize",
    "destabilized", "destabilizes", "destabilizing", "deviate",
    "deviated", "deviates", "deviating", "deviation", "deviations",
    "doubt", "doubted", "doubtful", "doubts", "dubious", "equivocal",
    "erratic", "erratically", "estimate", "estimated", "estimates",
    "estimating", "estimation", "estimations", "eventual", "eventually",
    "expect", "expectation", "expectations", "expected", "expecting",
    "expects", "expose", "exposed", "exposes", "exposing", "exposure",
    "exposures", "fairly", "fluctuate", "fluctuated", "fluctuates",
    "fluctuating", "fluctuation", "fluctuations", "hidden", "hinges",
    "hope", "hoped", "hopeful", "hopefully", "hopes", "hoping",
    "imprecise", "imprecision", "improbability", "improbable",
    "incompleteness", "indefinite", "indefinitely", "indefiniteness",
    "indeterminable", "indeterminate", "inexact", "inexactness",
    "instabilities", "instability", "intangible", "intangibles",
    "likelihood", "likely", "may", "maybe", "might", "nearly",
    "nonassessable", "occasionally", "ordinarily", "pending", "perhaps",
    "possibility", "possible", "possibly", "precaution", "precautionary",
    "precautions", "predict", "predictability", "predicted", "predicting",
    "prediction", "predictions", "predicts", "preliminarily", "preliminary",
    "presumably", "presume", "presumed", "presumes", "presuming",
    "presumption", "presumptions", "probabilistic", "probabilities",
    "probability", "probable", "probably", "random", "randomize",
    "randomized", "randomizes", "randomizing", "randomly", "randomness",
    "reassess", "reassessed", "reassesses", "reassessing", "reassessment",
    "reassessments", "recalculate", "recalculated", "recalculates",
    "recalculating", "recalculation", "recalculations", "reconsider",
    "reconsidered", "reconsidering", "reconsiders", "reestimate",
    "reestimated", "reestimates", "reestimating", "reexamine",
    "reexamined", "reexamines", "reexamining", "reinterpret",
    "reinterpretation", "reinterpretations", "reinterpreted",
    "reinterpreting", "reinterprets", "revise", "revised", "revises",
    "revising", "revision", "revisions", "risk", "risked", "riskier",
    "riskiest", "riskiness", "risking", "risks", "risky", "roughly",
    "rumors", "seems", "seldom", "somewhat", "sometimes", "somewhere",
    "speculate", "speculated", "speculates", "speculating", "speculation",
    "speculations", "speculative", "speculatively", "sporadic",
    "sporadically", "sudden", "suddenly", "suggest", "suggested",
    "suggesting", "suggestion", "suggestions", "suggests", "susceptibility",
    "tend", "tended", "tendencies", "tendency", "tending", "tends",
    "tentative", "tentatively", "turbulence", "uncertain", "uncertainly",
    "uncertainties", "uncertainty", "unclear", "unconfirmed",
    "undecided", "undefined", "undesignated", "undetectable",
    "undeterminable", "undetermined", "unexpected", "unexpectedly",
    "unfamiliar", "unforeseen", "unknown", "unknowns", "unobservable",
    "unpredictability", "unpredictable", "unpredictably", "unproven",
    "unquantifiable", "unquantified", "unreliable", "unsettled",
    "unspecified", "untested", "unusual", "unusually", "vague",
    "vaguely", "vagueness", "variability", "variable", "variables",
    "variably", "variance", "variances", "variant", "variants",
    "variation", "variations", "varied", "varies", "vary", "varying",
    "volatile", "volatilities", "volatility"
}

# Litigious words - indicate legal/litigation concerns
# Associated with higher litigation risk
LITIGIOUS_WORDS = {
    "abovementioned", "acquit", "acquits", "acquittal", "acquittals",
    "acquitted", "acquitting", "adjudge", "adjudged", "adjudges",
    "adjudging", "adjudicate", "adjudicated", "adjudicates",
    "adjudicating", "adjudication", "adjudications", "affiants",
    "affidavit", "affidavits", "alleged", "allegedly", "alleges",
    "alleging", "antitrust", "appeal", "appealed", "appealing",
    "appeals", "appellate", "arbitral", "arbitrate", "arbitrated",
    "arbitrates", "arbitrating", "arbitration", "arbitrations",
    "arbitrator", "arbitrators", "arraign", "arraigned", "arraigning",
    "arraignment", "arraignments", "arraigns", "arrest", "arrested",
    "arresting", "arrests", "attorney", "attorneys", "bail",
    "bailiff", "bailiffs", "bails", "bankrupt", "bankruptcies",
    "bankruptcy", "bankrupted", "bankrupting", "bankrupts", "bar",
    "barred", "barring", "bars", "bench", "benches", "breach",
    "breached", "breaches", "breaching", "bribe", "bribed", "briberies",
    "bribery", "bribes", "bribing", "case", "caseload", "caseloads",
    "cases", "claimant", "claimants", "claimed", "claiming", "claims",
    "class", "collude", "colluded", "colludes", "colluding", "collusion",
    "collusions", "collusive", "complain", "complainant", "complainants",
    "complained", "complaining", "complains", "complaint", "complaints",
    "compulsion", "compulsory", "confiscate", "confiscated", "confiscates",
    "confiscating", "confiscation", "confiscations", "conspiracy",
    "conspirator", "conspirators", "conspire", "conspired", "conspires",
    "conspiring", "contempt", "contempts", "convict", "convicted",
    "convicting", "conviction", "convictions", "convicts", "counsel",
    "counseled", "counseling", "counsels", "counterclaim", "counterclaimed",
    "counterclaiming", "counterclaims", "court", "courts", "crime",
    "crimes", "criminal", "criminally", "criminals", "crossclaim",
    "crossclaims", "culpability", "culpable", "damages", "decree",
    "decreed", "decreeing", "decrees", "defendant", "defendants",
    "defraud", "defrauded", "defrauding", "defrauds", "deposition",
    "depositions", "discovery", "docket", "docketed", "docketing",
    "dockets", "embezzle", "embezzled", "embezzlement", "embezzlements",
    "embezzler", "embezzlers", "embezzles", "embezzling", "encroach",
    "encroached", "encroaches", "encroaching", "encroachment",
    "encroachments", "enforce", "enforceability", "enforceable",
    "enforced", "enforcement", "enforcements", "enforces", "enforcing",
    "enjoin", "enjoined", "enjoining", "enjoins", "extradite",
    "extradited", "extradites", "extraditing", "extradition",
    "extraditions", "felonies", "felonious", "felony", "file",
    "filed", "files", "filing", "filings", "fine", "fined", "fines",
    "fining", "fraud", "frauds", "fraudulent", "fraudulently", "grievance",
    "grievances", "guilt", "guiltier", "guiltiest", "guilty", "hear",
    "heard", "hearing", "hearings", "hears", "illegal", "illegalities",
    "illegality", "illegally", "illegals", "imprison", "imprisoned",
    "imprisoning", "imprisonment", "imprisonments", "imprisons",
    "incarcerate", "incarcerated", "incarcerates", "incarcerating",
    "incarceration", "incarcerations", "indict", "indicted", "indictees",
    "indicting", "indictment", "indictments", "indicts", "infraction",
    "infractions", "infringe", "infringed", "infringement", "infringements",
    "infringer", "infringers", "infringes", "infringing", "injunction",
    "injunctions", "injunctive", "innocence", "innocent", "innocently",
    "jail", "jailed", "jailing", "jails", "judge", "judged", "judges",
    "judging", "judgment", "judgments", "judicial", "judicially",
    "juries", "jurisdiction", "jurisdictional", "jurisdictionally",
    "jurisdictions", "juror", "jurors", "jury", "justice", "justices",
    "law", "laws", "lawsuit", "lawsuits", "lawyer", "lawyers", "legal",
    "legality", "legally", "legislate", "legislated", "legislates",
    "legislating", "legislation", "legislations", "legislative",
    "legislatively", "legislator", "legislators", "legislature",
    "legislatures", "libel", "libeled", "libeling", "libelous", "libels",
    "litigant", "litigants", "litigate", "litigated", "litigates",
    "litigating", "litigation", "litigations", "magistrate", "magistrates",
    "mediate", "mediated", "mediates", "mediating", "mediation",
    "mediations", "mediator", "mediators", "misappropriate",
    "misappropriated", "misappropriates", "misappropriating",
    "misappropriation", "misappropriations", "misdemeanor", "misdemeanors",
    "misrepresent", "misrepresentation", "misrepresentations",
    "misrepresented", "misrepresenting", "misrepresents", "mistrial",
    "mistrials", "offence", "offences", "offend", "offended", "offender",
    "offenders", "offending", "offends", "offense", "offenses", "offenses",
    "overrule", "overruled", "overrules", "overruling", "parole",
    "paroled", "parolee", "parolees", "paroles", "paroling", "penalties",
    "penalty", "penalize", "penalized", "penalizes", "penalizing",
    "perjure", "perjured", "perjurer", "perjurers", "perjures",
    "perjuries", "perjuring", "perjury", "petition", "petitioned",
    "petitioner", "petitioners", "petitioning", "petitions", "plaintiff",
    "plaintiffs", "plea", "plead", "pleaded", "pleading", "pleadings",
    "pleads", "pleas", "pled", "police", "policed", "polices", "policing",
    "prison", "prisoner", "prisoners", "prisons", "probation",
    "probationary", "probations", "prosecute", "prosecuted", "prosecutes",
    "prosecuting", "prosecution", "prosecutions", "prosecutor",
    "prosecutors", "punish", "punished", "punishes", "punishing",
    "punishment", "punishments", "punitive", "racketeering", "recover",
    "recoverable", "recovered", "recoveries", "recovering", "recovers",
    "recovery", "rectification", "rectifications", "rectified",
    "rectifies", "rectify", "rectifying", "redress", "redressed",
    "redresses", "redressing", "remedial", "remediate", "remediated",
    "remediates", "remediating", "remediation", "remediations", "remedied",
    "remedies", "remedy", "remedying", "remunerate", "remunerated",
    "remunerates", "remunerating", "restitution", "restitutions",
    "restrain", "restrained", "restraining", "restrains", "restraint",
    "restraints", "retrial", "retrials", "ruling", "rulings", "sanction",
    "sanctioned", "sanctioning", "sanctions", "sentence", "sentenced",
    "sentences", "sentencing", "settle", "settled", "settlement",
    "settlements", "settles", "settling", "subpoena", "subpoenaed",
    "subpoenaing", "subpoenas", "sue", "sued", "sueing", "sues",
    "suing", "suit", "suits", "summon", "summoned", "summoning",
    "summons", "summonses", "testify", "testimony", "tort", "tortious",
    "torts", "trial", "trials", "tribunal", "tribunals", "verdict",
    "verdicts", "violate", "violated", "violates", "violating",
    "violation", "violations", "violator", "violators", "warrant",
    "warranted", "warranties", "warranting", "warrants", "witness",
    "witnessed", "witnesses", "witnessing", "writ", "writs", "wrongdoing",
    "wrongdoings", "wrongful", "wrongfully"
}

# Constraining words - indicate limitations or obligations
# Associated with operational constraints
CONSTRAINING_WORDS = {
    "abide", "abiding", "bound", "bounded", "commit", "commitment",
    "commitments", "commits", "committed", "committing", "compel",
    "compelled", "compelling", "compels", "comply", "complying",
    "condition", "conditioned", "conditions", "confine", "confined",
    "confinement", "confinements", "confines", "confining", "constrain",
    "constrained", "constraining", "constrains", "constraint",
    "constraints", "curb", "curbed", "curbing", "curbs", "curtail",
    "curtailed", "curtailing", "curtailment", "curtailments", "curtails",
    "forbid", "forbidden", "forbidding", "forbids", "force", "forced",
    "forces", "forcing", "hamper", "hampered", "hampering", "hampers",
    "hinder", "hindered", "hindering", "hinders", "impair", "impaired",
    "impairing", "impairment", "impairments", "impairs", "impede",
    "impeded", "impedes", "impeding", "inhibit", "inhibited", "inhibiting",
    "inhibition", "inhibitions", "inhibits", "limit", "limitation",
    "limitations", "limited", "limiting", "limits", "mandate", "mandated",
    "mandates", "mandating", "mandatorily", "mandatory", "must",
    "necessitate", "necessitated", "necessitates", "necessitating",
    "necessities", "necessity", "noncompliance", "noncompliances",
    "obligate", "obligated", "obligates", "obligating", "obligation",
    "obligations", "obligatorily", "obligatory", "preclude", "precluded",
    "precludes", "precluding", "preclusion", "preclusions", "prevent",
    "prevented", "preventing", "prevention", "preventions", "preventive",
    "prevents", "prohibit", "prohibited", "prohibiting", "prohibition",
    "prohibitions", "prohibitive", "prohibitively", "prohibits", "require",
    "required", "requirement", "requirements", "requires", "requiring",
    "requisite", "requisites", "restrain", "restrained", "restraining",
    "restrains", "restraint", "restraints", "restrict", "restricted",
    "restricting", "restriction", "restrictions", "restrictive",
    "restrictively", "restricts", "shall", "should"
}

# Weak modal words - indicate hedging/uncertainty in management forecasts
# Higher usage associated with earnings restatements
WEAK_MODAL_WORDS = {
    "almost", "apparently", "appear", "appeared", "appearing", "appears",
    "could", "depend", "depended", "dependence", "dependencies",
    "dependency", "dependent", "depending", "depends", "may", "maybe",
    "might", "nearly", "occasionally", "perhaps", "possible", "possibly",
    "seldom", "sometimes", "somewhat", "suggest", "suggested", "suggesting",
    "suggestion", "suggestions", "suggests"
}

# Strong modal words - indicate management certainty
# Higher usage associated with confidence in forecasts
STRONG_MODAL_WORDS = {
    "always", "best", "clearly", "definitely", "definitive", "definitively",
    "highest", "must", "never", "shall", "strongest", "will"
}

# Negative words (LM financial negative - different from general negative)
NEGATIVE_WORDS = {
    "abandon", "abandoned", "abandoning", "abandonment", "abandonments",
    "abandons", "abdicate", "abdicated", "abdicates", "abdicating",
    "abdication", "abdications", "aberrant", "aberration", "aberrational",
    "aberrations", "abetting", "abnormal", "abnormalities", "abnormality",
    "abnormally", "abolish", "abolished", "abolishes", "abolishing",
    "abolition", "abolitions", "abrupt", "abruptly", "abruptness",
    "absence", "absences", "absent", "abuse", "abused", "abuses",
    "abusing", "abusive", "abusively", "abusiveness", "accident",
    "accidental", "accidentally", "accidents", "accusation", "accusations",
    "accuse", "accused", "accuses", "accusing", "acquiesce", "acquiesced",
    "acquiesces", "acquiescing", "adulterate", "adulterated", "adulterates",
    "adulterating", "adulteration", "adulterations", "adversarial",
    "adversaries", "adversary", "adverse", "adversely", "adversities",
    "adversity", "aftermath", "aftermaths", "against", "aggravate",
    "aggravated", "aggravates", "aggravating", "aggravation", "aggravations",
    "alarmed", "alarming", "alarmingly", "alienate", "alienated",
    "alienates", "alienating", "alienation", "alienations", "allegation",
    "allegations"
    # ... truncated for brevity - full list has ~2,300 words
}

# Positive words (LM financial positive)
POSITIVE_WORDS = {
    "able", "abundance", "abundant", "acclaim", "acclaimed", "accomplish",
    "accomplished", "accomplishes", "accomplishing", "accomplishment",
    "accomplishments", "achieve", "achieved", "achievement", "achievements",
    "achieves", "achieving", "adequacy", "adequate", "adequately",
    "advantage", "advantaged", "advantageous", "advantageously", "advantages",
    "alliance", "alliances", "assure", "assured", "assures", "assuring",
    "attain", "attained", "attaining", "attainment", "attainments",
    "attains", "attractive", "attractively", "attractiveness", "beautiful",
    "beautifully", "beneficial", "beneficially", "benefit", "benefited",
    "benefiting", "benefits", "benefitted", "benefitting", "best",
    "better", "bolster", "bolstered", "bolstering", "bolsters", "boom",
    "boomed", "booming", "booms", "boost", "boosted", "boosting", "boosts",
    "breakthrough", "breakthroughs", "brilliant", "brilliantly"
    # ... truncated for brevity - full list has ~350 words
}


def count_word_category(text: str, word_set: set) -> int:
    """
    Count occurrences of words from a category in text.

    :param text: Input text (will be lowercased)
    :param word_set: Set of words to count
    :return: Total count of matching words
    """
    words = text.lower().split()
    return sum(1 for word in words if word.strip('.,!?;:"\'()[]{}') in word_set)


def compute_word_ratios(text: str) -> dict:
    """
    Compute all Loughran-McDonald word ratios for a text.

    :param text: Input text
    :return: Dict with word counts and ratios
    """
    words = text.lower().split()
    total_words = len(words)

    if total_words == 0:
        return {
            'word_count': 0,
            'uncertainty_count': 0,
            'uncertainty_ratio': 0.0,
            'litigious_count': 0,
            'litigious_ratio': 0.0,
            'constraining_count': 0,
            'constraining_ratio': 0.0,
            'weak_modal_count': 0,
            'weak_modal_ratio': 0.0,
            'strong_modal_count': 0,
            'strong_modal_ratio': 0.0,
        }

    # Clean words (remove punctuation)
    clean_words = [w.strip('.,!?;:"\'()[]{}') for w in words]

    uncertainty_count = sum(1 for w in clean_words if w in UNCERTAINTY_WORDS)
    litigious_count = sum(1 for w in clean_words if w in LITIGIOUS_WORDS)
    constraining_count = sum(1 for w in clean_words if w in CONSTRAINING_WORDS)
    weak_modal_count = sum(1 for w in clean_words if w in WEAK_MODAL_WORDS)
    strong_modal_count = sum(1 for w in clean_words if w in STRONG_MODAL_WORDS)

    return {
        'word_count': total_words,
        'uncertainty_count': uncertainty_count,
        'uncertainty_ratio': round(uncertainty_count / total_words, 6),
        'litigious_count': litigious_count,
        'litigious_ratio': round(litigious_count / total_words, 6),
        'constraining_count': constraining_count,
        'constraining_ratio': round(constraining_count / total_words, 6),
        'weak_modal_count': weak_modal_count,
        'weak_modal_ratio': round(weak_modal_count / total_words, 6),
        'strong_modal_count': strong_modal_count,
        'strong_modal_ratio': round(strong_modal_count / total_words, 6),
    }
