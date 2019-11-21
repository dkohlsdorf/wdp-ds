import { Component, OnInit } from '@angular/core';
import { Encoding } from '../entities/encoding';
import { Http, Headers } from '@angular/http';
import { map } from "rxjs/operators";

@Component({
  selector: 'app-encoding',
  templateUrl: './encoding.component.html',
  styleUrls: ['./encoding.component.css']
})

export class EncodingComponent implements OnInit {

  encodings: Array<Encoding> = [];
  displayedColumns: string[] = ['year', 'encoding', 'behavior', 'lvl', 'key', 'spot_id'];

  constructor(private http: Http) { }

  get() {
    let url = "https://wdp-ds.appspot.com/wdp/encodings";
    let headers = new Headers();
    headers.set('Accept', 'text/json');
    this.http
      .get(url, { headers })
      .pipe(map(resp => resp.json()))
      .subscribe(
        encodings => {
          this.encodings = encodings;
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
