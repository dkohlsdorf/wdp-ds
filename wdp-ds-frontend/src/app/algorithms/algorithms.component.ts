import { Component, OnInit } from '@angular/core';
import {Http, Headers} from '@angular/http';
import {Algorithm} from '../entities/algorithm';
import { map } from "rxjs/operators";

@Component({
  selector: 'app-algorithms',
  templateUrl: './algorithms.component.html',
  styleUrls: ['./algorithms.component.css']
})
export class AlgorithmsComponent implements OnInit {

  constructor(private http: Http) { }

  algorithms: Array<Algorithm> = [];

  get() {
    let url = 'https://wdp-ds.appspot.com/wdp/ds/algorithms'
    let headers = new Headers();
    headers.set('Accept', 'text/json');
    this.http
      .get(url, {headers})
      .pipe(map(resp => resp.json()))
      .subscribe(
        algorithms => {
          this.algorithms = algorithms;
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
