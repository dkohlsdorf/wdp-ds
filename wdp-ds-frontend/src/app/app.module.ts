import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppComponent } from './app.component';
import { EncodingComponent } from './encoding/encoding.component';
import {HttpModule} from '@angular/http';
import { ApptopComponent } from './apptop/apptop.component'

@NgModule({
  declarations: [
    AppComponent,
    EncodingComponent,
    ApptopComponent
  ],
  imports: [
    BrowserModule,
    HttpModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
