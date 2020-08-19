import { Injectable, Query } from '@angular/core';
import { Client } from 'elasticsearch-browser';
import * as elasticsearch from 'elasticsearch-browser';
// documentation "https://www.elastic.co/guide/en/elasticsearch/client/javascript-api/16.x/index.html"

@Injectable({
  providedIn: 'root'
})
export class EsService {
  url: string;
  port: string;
  userName: string;
  passWord: string;
  index: string;
  field: string = "text_content";
  query: string;
  queryBody: any;
  serverType: string;
  sourceFields = ["text_content", "title"]
  assessor: string;
  private client: Client

  constructor() {

   }

   private connect() {
     if (this.serverType === "local") {
      this.client = new elasticsearch.Client({
        "host": this.url
        // "log": 'trace'
      })
     } else {
      this.client = new elasticsearch.Client({
        "host": this.url.slice(0, 8) + this.userName + ':' + this.passWord + '@' + this.url.slice(8)
       //  host: 'https://elastic:rc1YO7bVXdY4cQKBFeCH2jJe@3daa91de24df498ab204529d12ff570a.us-central1.gcp.cloud.es.io:9243/',
        // "log": 'trace'
      })
     }
   }

   addRelevance(value): any {
    return this.client.index(value);
   }

   isAvailable(): any {
     if (!this.client) {
       console.error("Client is not yet created! Please confirm ES Server Settings first!");
       return false;
     } else {
      return this.client.ping({
        requestTimeout: Infinity
      });
     }
   }

   getDocuments(): any {
     return this.client.search({
       "index": this.index,
       "body": this.queryBody,
       "filterPath": ['hits.hits', '_scroll_id'],
       "scroll": "10m"
     });
   }

   setQuery(query): void {
     this.query = query;
     this.queryBody = {
       "size": 10,
       "query": {
         "match": {
           [this.field]: this.query
         }
        },
        "_source": this.sourceFields
      };
    }

   getNextPage(scroll_id): any {
    return this.client.scroll({
      "scrollId": scroll_id,
      "scroll": "10m",
      "filterPath": ['hits.hits', '_scroll_id']
    });
   }

   setIndex(index): void {
     this.index = index;
   }

   setEsServer(url, userName, passWord, index, serverType, assessor) {
     this.index = index;
     this.serverType = serverType;
     if (serverType === "local") {
        this.url = url;
        console.log(this.url);
     } else {
       this.url = url;
       this.userName = userName;
       this.passWord = passWord;
       console.log(this.url);
     };
     this.assessor = assessor;
     this.connect();
   }
}
