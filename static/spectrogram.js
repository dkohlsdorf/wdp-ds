/**
   TODO:
     [3] Gap algorithm
**/


function euclid(x, y) {
    let n = x.length;
    let distance = 0.0;
    for(let i = 0; i < n; i++) {
	distance += Math.pow(x[i] - y[i], 2);
    }
    return Math.sqrt(distance);
}

function min3(x, y, z) {
    let min = x;
    if(y < min) min = y;
    if(z < min) min = z;
    return min;
}

function dtw(x, y) {
    let n = x.length;
    let m = y.length;
    let d = x[0].length;

    let dp =  Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0.0));

    dp[0][0] = 0.0; 
    for(let i = 1; i < n + 1; i++) {
	for(let j = 1; j < m + 1; j++) {
	    dp[i][j] = min3(
		dp[i - 1][j],
		dp[i][j - 1],
		dp[i-1][j - 1]
	    );
	    dp[i][j] += euclid(x[i - 1], y[j - 1]);
	}
    }
    return dp[n][m];    
}

function nn(sequences, _i, _j, dist_th) {
    let neighbors = []
    for(let i = 0; i < sequences.length; i++) {
	for(let j = 0; j < sequences[i].length; j++) {
	    if(i!= _i && j != _j) {
		let dist = dtw(sequences[i][j], distances[_i][_j]);
		console.log("Distance (${i}, ${j}), (${_i}, ${_j}) = ${dist}");
		if(dist < dist_th) {
		    neighbors.push([i, j]);
		}		
	    }
	}
    }
    return neighbors;
}


const NOISE = 0;
const MIN_REGION = 250;

function region_score(probs) {
    let score = Math.max(... probs.slice(1));
    let noise = probs[0] + 1e-6;
    return score / noise;
}

function regions(sequence, th) {
    let N = sequence.length;
    let regions = [];
    let start = -1;
    for(let t = 0; t < N; t++) {
	let snr = region_score(sequence[t])
	//console.log(snr, th);
	if(snr > th && start < 0) {
	    console.log(console.log(sequence[t]));
	    start = t;
	}
	if(snr < th && start >= 0) {
	    console.log("b");
	    let len = N - start;
	    if(len > MIN_REGION) {
		console.log("c");
		regions.push([start, t, snr]);
	    }
	    start = -1;
	}	
    }
    let len = N - start;
    if(len > MIN_REGION && start >= 0) {
	regions.push([start, N, -1]);
    }   
    return regions;
}

function largest_width(images) {
    let maxLen = 0;
    for(let image of images) {
	let id = image.id;
	let gaps = get_gaps(id);
	let w = image.width;
	for(let gap of gaps) {
	    w += gap.len;
	}
	maxLen = Math.max(w, maxLen);
    }
    return maxLen;
}

function get_gaps(id) {
    // TODO mock
    gaps = {
	"05291001_RegPCA110_3969000_6615000": [
	    //{start: 500, len: 1000},
	    //{start: 2500, len: 4000},
	    //{start: 12500, len: 2000},
	]
    };
    
    return gaps[id] ?? [];
}

function getMousePos(canvas, evt) {
    var rect = canvas.getBoundingClientRect();
    return {
        x: evt.clientX - rect.left,
        y: evt.clientY - rect.top
    };
}

function spectrograms(images, data, canvas) {
    const max_width = largest_width(images);
    const height = images[0].height;
    canvas.width = max_width;
    canvas.height = images.length * height;
    
    const context = canvas.getContext('2d');    

    let all_regions = [];
    let sequences = [];
    for(let i = 0; i < images.length; i++) {
	let id   = images[i].id;
	let seq  = data[id];
	let reg  = regions(seq, 150.0); 
	let local_sequences = [];
	for(let r of reg) {
	    local_sequences.push(seq.slice(r[0], r[1]));
	}
	sequences.push(local_sequences);
	all_regions.push(reg);
    }

    let neighbors = {};
    for(let i = 0; i < sequences.length; i++) {
	for(let j = 0; j < sequences.length[i]; j++) {
	    neighbors[(i,j)] = nn(sequences, i, j, dist_th);
	}
    }
    
    for(let i = 0; i < images.length; i++) {
	let id   = images[i].id;
	let reg  = all_regions[i];
	let gaps = get_gaps(id);
	
	let cur_img = 0;
	let cum_gap = 0;
	for(let gap of gaps) {
	    let cur_canvas = cur_img + cum_gap;
	    context.drawImage(images[i],
			      cur_img, 0, gap.start - cur_img, images[i].height,
			      cur_canvas, i * images[i].height, gap.start - cur_img, images[i].height);
	    context.fillStyle = "red";
	    context.fillText(cur_canvas, cur_canvas, i * images[i].height + 10);
	    context.fillText(cur_canvas + (gap.start - cur_img),
			     cur_canvas +  (gap.start - cur_img), i * images[i].height + 10);
	    cur_img = gap.start;
	    cum_gap += gap.len;
	}
	let cur_canvas = cur_img + cum_gap;
	context.fillText(cur_canvas, cur_canvas, i * images[i].height);
	context.drawImage(images[i],
			  cur_img, 0, images[i].width - cur_img, images[i].height,
			  cur_canvas, i * images[i].height, images[i].width - cur_img, images[i].height);		
	context.fillStyle = "red";
	context.fillText(cur_canvas, cur_canvas, i * images[i].height + 10);
	context.fillText(cur_canvas + (images[i].width - cur_img), cur_canvas + (images[i].width - cur_img) - 30, i * images[i].height + 10);


	for(let r of reg) {
	    context.fillStyle = "green";                                                         
	    context.globalAlpha = 0.2;
	    context.fillRect(r[0], i * images[i].height,r[1] - r[0], images[i].height);
	    context.globalAlpha = 1.0;
	    
	    console.log(id + " " + r[0] + " " + r[1] + " " + r[2]);
	}
	
	console.log('---');
    }
    canvas.addEventListener("click", function (evt) {
	const mousePos = getMousePos(canvas, evt);
	const spec_id = Math.floor(mousePos.y / height);
	let region_id = -1;
	for(let i = 0; i < all_regions[spec_id].length; i++) {
	    let region = all_regions[spec_id][i];
	    if(mousePos.x > region[0] && mousePos.x < region[1]) {
		region_id = i;
		break;
	    }
	}
	console.log(mousePos.x + ',' + mousePos.y + ": " + spec_id + " " + region_id);
    }, false);
}

function getData(url) {
    return fetch(url)
	.then(response=>{
	    return response.json()
	})
}
