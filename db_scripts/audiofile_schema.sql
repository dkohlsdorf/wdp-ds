DROP TABLE IF EXISTS wdp_ds.audio;
CREATE TABLE wdp_ds.audio (
    encoding     BIGINT NOT NULL,
    filename     VARCHAR(512),
    FOREIGN KEY (encoding)
    REFERENCES wdp_ds.encoding(encoding)
)      
;