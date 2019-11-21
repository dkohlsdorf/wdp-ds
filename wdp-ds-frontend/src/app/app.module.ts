import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppComponent } from './app.component';
import { EncodingComponent } from './encoding/encoding.component';
import {HttpModule} from '@angular/http';
import { ApptopComponent } from './apptop/apptop.component';
import { AlgorithmsComponent } from './algorithms/algorithms.component'
import { RouterModule, Routes } from '@angular/router';
import { MatIconModule, MatSidenavModule, MatMenuModule, MatToolbarModule, MatButtonModule, MatTableModule } from  '@angular/material';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { ClusterComponent } from './cluster/cluster.component';
import { MatListModule } from '@angular/material/list';
import { SpectrogramComponent } from './spectrogram/spectrogram.component';
import { CorrelationClusterFilesComponent } from './correlation-cluster-files/correlation-cluster-files.component';

const appRoutes: Routes = [
  { path: 'wdp-app/correlation/:cluster_name', component: CorrelationClusterFilesComponent},
  { path: 'wdp-app/spectrogram/:cluster_name/:asset_name', component: SpectrogramComponent},
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
    ClusterComponent,
    SpectrogramComponent,
    CorrelationClusterFilesComponent 
  ],
  imports: [
    BrowserModule,
    HttpModule,
    RouterModule.forRoot(appRoutes),
    BrowserAnimationsModule,
    MatToolbarModule,
    MatButtonModule,
    MatTableModule,
    MatMenuModule,
    MatListModule,
    MatIconModule,
    MatSidenavModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
