import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';

@Component({
  selector: 'app-cluster',
  templateUrl: './cluster.component.html',
  styleUrls: ['./cluster.component.css']
})
export class ClusterComponent implements OnInit {

  cluster_name: string;

  constructor(private route: ActivatedRoute) {     
    route.params.subscribe(      
      params => {
        this.cluster_name = params['cluster_name'];
      });
  }

  ngOnInit() {}

}
