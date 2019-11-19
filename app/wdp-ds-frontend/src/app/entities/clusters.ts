export interface Cluster {
    id:  bigint;
    encoding: bigint;
    filename: string;
    start: bigint;
    stop: bigint;
    algorithm: string;
    cluster_id: bigint;
    created_at: string;
}