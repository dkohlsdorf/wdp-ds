import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import {Http, Headers} from '@angular/http'
import { map } from "rxjs/operators";

@Component({
  selector: 'app-cluster',
  templateUrl: './cluster.component.html',
  styleUrls: ['./cluster.component.css']
})
export class ClusterComponent implements OnInit {

  cluster_name: string;
  files: Array<File> = []

  constructor(private route: ActivatedRoute, private http: Http) {     
    route.params.subscribe(      
      params => {
        this.cluster_name = params['cluster_name'];
      });
  }
  
  get() {
    let url = `/wdp/asset/cluster_files/${this.cluster_name}`;
    let headers = new Headers();
    headers.set('Accept', 'text/json');
    this.http
      .get(url, { headers })
      .pipe(map(resp => resp.json()))
      .subscribe(
        files => {
          this.files = files;
        },
        err => {
          console.error(err);
        }
      );
  }

  ngOnInit() {
    this.get();
  }

}
