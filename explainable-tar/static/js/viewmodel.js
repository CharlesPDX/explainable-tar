var ViewModel = function() {
    self = this;
    this.seedKeywords = ko.observable("");
    this.firstRun = ko.observable(true);
    this.currentDocument = ko.observable("");
    this.currentIndex = 0;
    this.documents = null;
    this.scores = null;
    this.documentIds = null;
    this.documentCategories = null;
    this.explanation = ko.observable("");
    this.relevanceScores = [];
    this.representations = ko.observableArray(["tf-idf", "word2vec", "glove"]);
    this.selectedRepresentation = ko.observable("");
    
    this.gettingData = ko.observable(false);
    this.relevanceEnabled = ko.computed(function(){return !self.gettingData() && !self.firstRun()});

    this.gettingExplanation = ko.observable(false);
    this.searching = ko.observable(false);

    this.seedSearch = function(){
        self.firstRun(false);
        self.gettingData(true);
        self.searching(true);
        var data = {"seedKeywords": self.seedKeywords(), "vectorizer": self.selectedRepresentation()};
        $.post("/search", data, function(returnedData) {
            self.gettingData(false);
            self.searching(false);
            console.log(returnedData);
            self.documents = returnedData["documents"];
            self.scores = returnedData["scores"];
            self.documentIds = returnedData["document_ids"];
            self.currentDocument(self.documents[self.currentIndex]);
        })
    };

    this.reset = function(){
        $.post("/reset", function(returnedData) {
            self.firstRun(true);
            self.selectedRepresentation("tf-idf")
            self.documents = null;
            self.scores = null;
            self.documentIds = null;
            self.currentDocument("");
            self.seedKeywords("");
            self.explanation("");
            self.relevanceScores = [];
        })
    };

    this.explain = function(){
        if (self.scores){
            self.explanation(`Document was ranked ${self.currentIndex + 1} based on the seed query using the BM25Okapi method, with a score of ${self.scores[self.currentIndex]}`);
            return;
        }
        self.gettingData(true);
        self.gettingExplanation(true);
        var explainData = JSON.stringify({documentId: self.documentIds[self.currentIndex]})
        $.ajax({
            url: "/explain",
            data : explainData,
            contentType : 'application/json',
            type : 'POST',
            success:  function(returnedData) {
                self.gettingData(false);
                self.gettingExplanation(false);
                console.log(returnedData);
                if (returnedData["explanation_type"] == "string"){
                    self.explanation(returnedData["explanation"])
                }
            }
        })
    };

    this.clear = function(){
        self.explanation("");
    };

    this.scoreRelevant = function(){
        self.scoreCore(1);
    };

    this.scoreNotRelevant = function(){
        self.scoreCore(0);
    };

    this.scoreCore = function(relevance){
        self.relevanceScores.push(relevance);
        self.currentIndex += 1;
        
        if (self.currentIndex >= self.documents.length){
            self.scores = null;
            self.currentIndex = 0;
            self.gettingData(true);
            self.searching(true);
            var scoreData = JSON.stringify({documentIds: self.documentIds, relevanceScores: self.relevanceScores})
            $.ajax({
                url: "/score",
                data : scoreData,
                contentType : 'application/json',
                type : 'POST',
                success:  function(returnedData) {
                    console.log(returnedData);
                    self.documents = returnedData["documents"];
                    self.documentIds = returnedData["document_ids"];
                    self.documentCategories = returnedData["doc_categories"];
                    self.currentDocument(self.documents[self.currentIndex]);
                    self.relevanceScores = [];
                    self.gettingData(false);
                    self.searching(false);
                }
            })
        }
        else{
            self.currentDocument(self.documents[self.currentIndex]);
            self.explanation("");
        }
    };
};
 
window.onload = (event) => {
    window.ActiveViewModel = new ViewModel();
    ko.applyBindings(window.ActiveViewModel);
  };
