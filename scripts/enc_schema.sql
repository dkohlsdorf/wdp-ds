DROP TABLE IF EXISTS wdp_ds.encoding;
CREATE TABLE wdp_ds.encoding (
       id           BIGINT PRIMARY KEY,
       YEAR         BIGINT,
       encoding     BIGINT,
       behavior     VARCHAR(32),
       activity_lvl DOUBLE,
       stenella_key VARCHAR(512),
       spot_id      VARCHAR(512)
)      
;

-- EXAMPLE
select spot_id from wdp_ds.encoding where INSTR(spot_id, ' VEGA ');
