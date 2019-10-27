DROP TABLE IF EXISTS wdp_ds.not_silent;
CREATE TABLE wdp_ds.not_silent (
       id           BIGINT PRIMARY KEY NOT NULL,
       encoding     BIGINT NOT NULL,
       filename     VARCHAR(512),
       start        BIGINT,
       stop         BIGINT,       
       FOREIGN KEY (encoding)
        REFERENCES wdp_ds.encoding(encoding)
)      
;
