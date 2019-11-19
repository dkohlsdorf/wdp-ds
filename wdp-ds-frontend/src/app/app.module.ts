import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppComponent } from './app.component';
import { EncodingComponent } from './encoding/encoding.component';
import {HttpModule} from '@angular/http'

@NgModule({
  declarations: [
    AppComponent,
    EncodingComponent
  ],
  imports: [
    BrowserModule,
    HttpModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
