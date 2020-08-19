import { Component, OnInit, Input } from '@angular/core';
import { EsService } from '../es.service';

@Component({
  selector: 'app-sidebar',
  templateUrl: './sidebar.component.html',
  styleUrls: ['./sidebar.component.css']
})

export class SidebarComponent implements OnInit {
  isConnected = false;
  status: string = "Unknown";
  hide = true;
  assessor:string = "Zhuocheng_Lin";

  url: string = "http://localhost:9200";
  userName: string = "Enter here";
  // passWord: string = "nRGUXlzD1f8kOT63iLehSG9a";
  passWord: string = "Enter here";
  index: string = "hw3";
  serverType: string = "local";

  constructor(private es: EsService) { }

  setEsServer(): void {
    if (this.url && this.index && this.serverType) {
      this.es.setEsServer(
        this.url,
        this.userName,
        this.passWord,
        this.index,
        this.serverType,
        this.assessor);

      this.es.isAvailable().then(
        () => {
          this.status = 'OK!';
          this.isConnected = true;
        },
        error => {
          this.status = 'OFFLINE!';
          this.isConnected = false;
          console.error('Server not online, check ES Server Settings', error);
        }
      )
    } else {
      console.error("Server Info Required!");
    }
  }

  ngOnInit(): void {

  }



}
