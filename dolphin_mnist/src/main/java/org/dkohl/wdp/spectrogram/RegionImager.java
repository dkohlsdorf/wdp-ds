package org.dkohl.wdp.spectrogram;

import java.awt.*;
import java.util.ArrayList;

public class RegionImager {

    private SpectrogramParams params;
    private ArrayList<Annotation> annotations;
    private AudioReader stream;
    private KeyMap keymap;

    public RegionImager(SpectrogramParams params, ArrayList<Annotation> annotations, AudioReader stream, KeyMap keymap) {
        this.params = params;
        this.annotations = annotations;
        this.stream = stream;
        this.keymap = keymap;
    }

    public void plot(Graphics2D g2d, int start, int w, double scaleW, int imageHeight, Color fill, Color border) {
        Color color = fill;
        Annotation match = Annotation.findAnnotation(annotations, params, stream, start, start + w);
        if(match != null) {
            int i = keymap.getKey(match.getAnnotation()) - 48;
            color = AnnotationColor.getColor(i, 0.2f);
        }
        g2d.setColor(color);
        g2d.fillRect((int) (start * scaleW), 0, (int) (w * scaleW), imageHeight);
        g2d.setColor(border);
        g2d.drawRect((int) (start * scaleW), 0, (int) (w * scaleW), imageHeight);
    }

}
