DROP TABLE IF EXISTS wdp_ds.not_silent;
CREATE TABLE wdp_ds.not_silent (
       id           BIGINT PRIMARY KEY NOT NULL,
       encoding     BIGINT NOT NULL,
       start        BIGINT,
       stop         BIGINT,       
       FOREIGN KEY (encoding)
        REFERENCES wdp_ds.encoding(encoding)
)      
;
