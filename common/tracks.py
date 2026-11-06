"""Definicja torów konkursu i walidator toru CLASSIC — NIE MODYFIKOWAĆ na własnych branchach.

Tory:
    "open"    — dowolne metody (np. gradient boosting, stacking, własne sieci, cokolwiek).
    "classic" — WYŁĄCZNIE modele z listy:
                regresja logistyczna, SVM, KNN, drzewa decyzyjne, Bagging, AdaBoost,
                Random Forest, Ensemble voting, sieci neuronowe.

Ograniczenie toru CLASSIC dotyczy MODELI (wszystkiego, co uczy się przewidywać).
Przetwarzanie danych i inżynieria cech są dowolne (pandas, imputacja, skalowanie, PCA,
encodery, ręcznie liczone cechy typu Elo/forma), o ile wewnątrz nie ukrywają
niedozwolonego modelu (np. SelectFromModel(GradientBoosting...) jest zabronione).

Użycie w torze CLASSIC — wywołaj na KAŻDYM modelu przed fit:
    from common.tracks import assert_classic
    model = make_pipeline(StandardScaler(), LogisticRegression())
    assert_classic(model)
"""

TRACKS = ("open", "classic")

# Dozwolone klasy estymatorów w torze CLASSIC (nazwy klas, sprawdzane wraz z modułem).
CLASSIC_ALLOWED = {
    # regresja logistyczna
    "sklearn.linear_model": {"LogisticRegression", "LogisticRegressionCV"},
    # SVM
    "sklearn.svm": {"SVC", "LinearSVC", "NuSVC", "SVR", "LinearSVR", "NuSVR"},
    # KNN
    "sklearn.neighbors": {"KNeighborsClassifier", "KNeighborsRegressor",
                          "RadiusNeighborsClassifier", "RadiusNeighborsRegressor"},
    # drzewa decyzyjne
    "sklearn.tree": {"DecisionTreeClassifier", "DecisionTreeRegressor"},
    # bagging, AdaBoost, Random Forest, voting
    "sklearn.ensemble": {"BaggingClassifier", "BaggingRegressor",
                         "AdaBoostClassifier", "AdaBoostRegressor",
                         "RandomForestClassifier", "RandomForestRegressor",
                         "VotingClassifier", "VotingRegressor"},
    # sieci neuronowe (sklearn)
    "sklearn.neural_network": {"MLPClassifier", "MLPRegressor"},
    # kalibracja prawdopodobieństw (post-processing modelu bazowego)
    "sklearn.calibration": {"CalibratedClassifierCV"},
}
# Sieci neuronowe z bibliotek DL są dozwolone w całości (moduły tych pakietów).
CLASSIC_ALLOWED_PACKAGES = ("torch", "skorch", "keras", "tensorflow")


def _is_model(obj) -> bool:
    """Czy obiekt jest predyktorem (klasyfikator/regresor), a nie transformatorem."""
    try:
        from sklearn.base import is_classifier, is_regressor
    except ImportError:  # pragma: no cover
        return False
    return is_classifier(obj) or is_regressor(obj)


def _allowed(obj) -> bool:
    cls = type(obj)
    module, name = cls.__module__, cls.__name__
    if module.split(".")[0] in CLASSIC_ALLOWED_PACKAGES:
        return True
    return any(module.startswith(m) and name in names for m, names in CLASSIC_ALLOWED.items())


def _walk(obj, seen):
    if id(obj) in seen:
        return
    seen.add(id(obj))
    yield obj
    children = []
    if hasattr(obj, "get_params"):
        try:
            children = list(obj.get_params(deep=False).values())
        except Exception:
            children = []
    elif isinstance(obj, (list, tuple)):
        children = list(obj)
    elif isinstance(obj, dict):
        children = list(obj.values())
    for child in children:
        if hasattr(child, "get_params") or isinstance(child, (list, tuple, dict)):
            yield from _walk(child, seen)


def classic_violations(model) -> list[str]:
    """Lista niedozwolonych modeli znalezionych w (zagnieżdżonym) estymatorze."""
    bad = []
    for obj in _walk(model, set()):
        if hasattr(obj, "get_params") and _is_model(obj) and not _allowed(obj):
            name = type(obj).__name__
            # Pipeline/GridSearchCV itp. "dziedziczą" typ po modelu końcowym — to nie są modele same w sobie.
            if type(obj).__module__.startswith(("sklearn.pipeline", "sklearn.model_selection", "sklearn.compose")):
                continue
            bad.append(f"{type(obj).__module__}.{name}")
    return bad


def assert_classic(model) -> None:
    """Rzuca ValueError, jeśli model zawiera cokolwiek spoza listy toru CLASSIC."""
    bad = classic_violations(model)
    if bad:
        raise ValueError(
            "Tor CLASSIC: niedozwolone modele: " + ", ".join(sorted(set(bad)))
            + ". Dozwolone: LogReg, SVM, KNN, drzewa, Bagging, AdaBoost, RandomForest, Voting, sieci neuronowe."
        )
