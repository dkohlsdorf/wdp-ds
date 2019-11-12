DROP TABLE IF EXISTS wdp_ds.clustering_results;
CREATE TABLE wdp_ds.clustering_results (
       id           BIGINT PRIMARY KEY NOT NULL,
       encoding     BIGINT NOT NULL,
       filename     VARCHAR(512),
       start        BIGINT,
       stop         BIGINT,   
       algorithm    VARCHAR(64),
       cluster_id   BIGINT,
       created_at   TIMESTAMP,
       FOREIGN KEY (encoding)
        REFERENCES wdp_ds.encoding(encoding)
)      
;
