import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppComponent } from './app.component';
import { EncodingComponent } from './encoding/encoding.component';
import {HttpModule} from '@angular/http';
import { ApptopComponent } from './apptop/apptop.component';
import { AlgorithmsComponent } from './algorithms/algorithms.component'
import { RouterModule, Routes } from '@angular/router';
import { MatMenuModule, MatToolbarModule, MatButtonModule, MatTableModule } from  '@angular/material';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { ClusterComponent } from './cluster/cluster.component';

const appRoutes: Routes = [
  { path: 'wdp-app/algorithm/:cluster_name', component: ClusterComponent},
  { path: 'wdp-app/encodings', component: EncodingComponent},
  { path: '',
    redirectTo: '/wdp-app/encodings',
    pathMatch: 'full'
  },
];

@NgModule({
  declarations: [
    AppComponent,
    EncodingComponent,
    ApptopComponent,
    AlgorithmsComponent,
    ClusterComponent    
  ],
  imports: [
    BrowserModule,
    HttpModule,
    RouterModule.forRoot(appRoutes),
    BrowserAnimationsModule,
    MatToolbarModule,
    MatButtonModule,
    MatTableModule,
    MatMenuModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
