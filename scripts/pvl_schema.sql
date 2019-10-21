DROP TABLE IF EXISTS wdp_ds.pvl;
CREATE TABLE wdp_ds.pvl (
       id          BIGINT PRIMARY KEY,
       encoding    BIGINT,
       species     VARCHAR(256),
       YEAR        BIGINT,
       sound_type  VARCHAR(8),
       context     VARCHAR(32),
       description VARCHAR(512),
       spot_id     VARCHAR(8),
       timecode    VARCHAR(32)
)
;

-- Example
SELECT * FROM wdp_ds.pvl WHERE FIND_IN_SET('COPPER', description) AND FIND_IN_SET('WH', sound_type) AND YEAR = 2008;
