DROP TABLE IF EXISTS wdp_ds.pvl;
CREATE TABLE wdp_ds.pvl (
       id          BIGINT PRIMARY KEY,
       YEAR        BIGINT,
       encoding    BIGINT NOT NULL,
       context     VARCHAR(32),
       sound_type  VARCHAR(8),
       species     VARCHAR(256),
       spot_id     VARCHAR(8),
       timecode    VARCHAR(32),
       description VARCHAR(512),
       FOREIGN KEY (encoding)
       REFERENCES encoding(encoding)
)
;

-- Example
SELECT
	y.year, 
	y.encoding,
	x.timecode,
	x.description
FROM
	wdp_ds.pvl      x
JOIN
	wdp_ds.encoding y
ON
	x.encoding = y.encoding 
WHERE
	INSTR(y.spot_id, ' BISHU ')    AND
	INSTR(x.description, ' SWIM ') AND
	x.YEAR = 2008
ORDER BY
	y.encoding
;
