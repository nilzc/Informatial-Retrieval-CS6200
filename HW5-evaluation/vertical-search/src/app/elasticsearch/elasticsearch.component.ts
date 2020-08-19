import { Component, OnInit, ChangeDetectorRef, ChangeDetectionStrategy } from '@angular/core';
import { EsService } from '../es.service';
import { MatSnackBar } from '@angular/material/snack-bar';

@Component({
  selector: 'app-elasticsearch',
  templateUrl: './elasticsearch.component.html',
  styleUrls: ['./elasticsearch.component.css']
})

export class ElasticsearchComponent implements OnInit {
  query: string;
  scroll_id: string;
  result: any;
  nextPage = true;
  notFound = false
  assessor:string;

  notAvailable = false;
  relevance = [
    {"value": 0, "viewValue": "Non-Relevant"},
    {"value": 1, "viewValue": "Relevant"},
    {"value": 2, "viewValue": "Very Relevant"}
  ];
  docRelevance = null;

  relevanceList = {};

  reset() {
    this.relevanceList = {}
  }

  setRelevance(rel) {
    this.docRelevance = rel;
  }

  updateRelevance(doc_id) {
    if (this.docRelevance === null) {
      this._snackBar.open(
        "Please Select/Change a Relevance Level!",
        "Got it",
        {
          "duration": 3000,
          "horizontalPosition": "end"});
      console.error("New Relevance needed!");
      return
    };
    this.relevanceList[doc_id] = this.docRelevance;
    this._snackBar.open(
      `Assessed Docs: ${Object.keys(this.relevanceList).length}`,
      "Got it",
      {
        "duration": 3000,
        "horizontalPosition": "end"});
    console.log(this.relevanceList);
    if (Object.keys(this.relevanceList).length === 200) {
      this.assessor = this.es.assessor;
      console.log(this.query)
      var id = null;
      if (this.query === "College of Cardinals") {
        id = this.assessor + ", 151901"
      } else if (this.query === "Ten Commandments") {
        id = this.assessor + ", 151902"
      } else if (this.query === "Recent Popes") {
        id = this.assessor + ", 151903"
      };
      if (id === null) {
        alert("Please enter the correct query!");
        return
      }
      this.es.addRelevance({
        index: "hw5",
        id: id,
        body: {
          relevance: this.relevanceList
        }
      }).then((result) => {
        console.log(result);
        alert("Docs added, see log for more info");
      }, error => {
        alert("Something went wrong, see log for more info");
        console.error(error);
      });
    };
    this.docRelevance = null;
  }

  search(query) {
    if (this.es.isAvailable()) {
      this.notAvailable = false;
      this.es.setQuery(query);
      this.es.getDocuments().then(
        res => {
          this.result = res.hits.hits;
          this.scroll_id = res._scroll_id;
          this.notFound = false;
          if (this.result.length === 0) {
            this.nextPage = false;
            this.notFound = true;
          }
          console.log("Docs retrieved!");
        },
        error => {
          console.error(error);
          this.notFound = true;}
      );
    } else {
      this.notAvailable = true;
      this._snackBar.open(
        "Please confirm ES Server Settings first!",
        "Got it",
        {
          "duration": 3000,
          "horizontalPosition": "end"});
    }
  }

  getNextPage() {
    this.es.getNextPage(this.scroll_id).then(
      res => {
        this.result = res.hits.hits;
        this.scroll_id = res._scroll_id;
        console.log("Docs retrieved!");
      },
      error => console.error(error)
    );
    window.scrollTo(0, 0);
  }

  constructor(private es: EsService, private _snackBar: MatSnackBar) {
   }

  ngOnInit(): void {}
}
