.headers on
.mode column

ATTACH 'HPML-proj/output/nsys_ddp_4gpu.sqlite' AS ddp;
ATTACH 'HPML-proj/output/nsys_fsdp_4gpu.sqlite' AS fsdp;
ATTACH 'HPML-proj/output/nsys_ddp_4gpu_peft.sqlite' AS ddp_peft;
ATTACH 'HPML-proj/output/nsys_fsdp_4gpu_peft.sqlite' AS fsdp_peft;

SELECT 'Section 1: Event volume overview' AS section;
WITH counts AS (
  SELECT 'ddp' AS method,
         (SELECT COUNT(*) FROM ddp.CUPTI_ACTIVITY_KIND_KERNEL) AS kernel_events,
         (SELECT COUNT(*) FROM ddp.CUPTI_ACTIVITY_KIND_RUNTIME) AS runtime_events,
         (SELECT COUNT(*) FROM ddp.CUPTI_ACTIVITY_KIND_MEMCPY) AS memcpy_events,
         (SELECT COUNT(*) FROM ddp.NVTX_EVENTS WHERE end IS NOT NULL) AS nvtx_events
  UNION ALL
  SELECT 'fsdp',
         (SELECT COUNT(*) FROM fsdp.CUPTI_ACTIVITY_KIND_KERNEL),
         (SELECT COUNT(*) FROM fsdp.CUPTI_ACTIVITY_KIND_RUNTIME),
         (SELECT COUNT(*) FROM fsdp.CUPTI_ACTIVITY_KIND_MEMCPY),
         (SELECT COUNT(*) FROM fsdp.NVTX_EVENTS WHERE end IS NOT NULL)
)
SELECT * FROM counts;

SELECT '' AS '';
SELECT 'Section 2: NVTX training phase summary' AS section;
WITH phase_totals AS (
  SELECT 'ddp' AS method, COALESCE(n.text, s.value) AS phase, SUM(n.end - n.start) / 1e9 AS total_s
  FROM ddp.NVTX_EVENTS n
  LEFT JOIN ddp.StringIds s ON n.textId = s.id
  WHERE n.end IS NOT NULL AND COALESCE(n.text, s.value) IN ('forward_loss', 'backward', 'optimizer_step')
  GROUP BY phase
  UNION ALL
  SELECT 'fsdp' AS method, COALESCE(n.text, s.value) AS phase, SUM(n.end - n.start) / 1e9 AS total_s
  FROM fsdp.NVTX_EVENTS n
  LEFT JOIN fsdp.StringIds s ON n.textId = s.id
  WHERE n.end IS NOT NULL AND COALESCE(n.text, s.value) IN ('forward_loss', 'backward', 'optimizer_step')
  GROUP BY phase
), method_totals AS (
  SELECT method, SUM(total_s) AS method_total_s FROM phase_totals GROUP BY method
)
SELECT p.method,
       p.phase,
       ROUND(p.total_s, 3) AS total_s,
       ROUND(100.0 * p.total_s / m.method_total_s, 2) AS pct_of_named_train_phases
FROM phase_totals p
JOIN method_totals m USING(method)
ORDER BY p.method, p.total_s DESC;

SELECT '' AS '';
SELECT 'Section 3: NCCL GPU kernel breakdown' AS section;
WITH nccl AS (
  SELECT 'ddp' AS method,
         s.value AS kernel_name,
         SUM(k.end - k.start) / 1e9 AS total_s
  FROM ddp.CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN ddp.StringIds s ON k.shortName = s.id
  WHERE s.value LIKE 'ncclDevKernel%'
  GROUP BY s.value
  UNION ALL
  SELECT 'fsdp' AS method,
         s.value AS kernel_name,
         SUM(k.end - k.start) / 1e9 AS total_s
  FROM fsdp.CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN fsdp.StringIds s ON k.shortName = s.id
  WHERE s.value LIKE 'ncclDevKernel%'
  GROUP BY s.value
)
SELECT method, kernel_name, ROUND(total_s, 3) AS total_s
FROM nccl
ORDER BY method, total_s DESC;

SELECT '' AS '';
SELECT 'Section 4: Communication-kernel share of total GPU kernel time' AS section;
WITH all_k AS (
  SELECT 'ddp' AS method, SUM(end - start) / 1e9 AS total_kernel_s FROM ddp.CUPTI_ACTIVITY_KIND_KERNEL
  UNION ALL
  SELECT 'fsdp', SUM(end - start) / 1e9 FROM fsdp.CUPTI_ACTIVITY_KIND_KERNEL
), nccl_k AS (
  SELECT 'ddp' AS method, SUM(k.end - k.start) / 1e9 AS nccl_kernel_s
  FROM ddp.CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN ddp.StringIds s ON k.shortName = s.id
  WHERE s.value LIKE 'ncclDevKernel%'
  UNION ALL
  SELECT 'fsdp' AS method, SUM(k.end - k.start) / 1e9 AS nccl_kernel_s
  FROM fsdp.CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN fsdp.StringIds s ON k.shortName = s.id
  WHERE s.value LIKE 'ncclDevKernel%'
)
SELECT a.method,
       ROUND(a.total_kernel_s, 3) AS total_kernel_s,
       ROUND(n.nccl_kernel_s, 3) AS nccl_kernel_s,
       ROUND(100.0 * n.nccl_kernel_s / a.total_kernel_s, 2) AS nccl_pct_of_total_kernel_time
FROM all_k a
JOIN nccl_k n USING(method);

SELECT '' AS '';
SELECT 'Section 5: Device-to-device memcpy volume' AS section;
WITH d2d AS (
  SELECT 'ddp' AS method,
         SUM(bytes) / 1024.0 / 1024.0 AS total_mb,
         COUNT(*) AS copies,
         AVG(bytes) / 1024.0 / 1024.0 AS avg_mb
  FROM ddp.CUPTI_ACTIVITY_KIND_MEMCPY m
  JOIN ddp.ENUM_CUDA_MEM_KIND sk ON m.srcKind = sk.id
  JOIN ddp.ENUM_CUDA_MEM_KIND dk ON m.dstKind = dk.id
  WHERE sk.label = 'Device' AND dk.label = 'Device'
  UNION ALL
  SELECT 'fsdp' AS method,
         SUM(bytes) / 1024.0 / 1024.0 AS total_mb,
         COUNT(*) AS copies,
         AVG(bytes) / 1024.0 / 1024.0 AS avg_mb
  FROM fsdp.CUPTI_ACTIVITY_KIND_MEMCPY m
  JOIN fsdp.ENUM_CUDA_MEM_KIND sk ON m.srcKind = sk.id
  JOIN fsdp.ENUM_CUDA_MEM_KIND dk ON m.dstKind = dk.id
  WHERE sk.label = 'Device' AND dk.label = 'Device'
)
SELECT method,
       ROUND(total_mb, 1) AS total_mb,
       copies,
       ROUND(avg_mb, 3) AS avg_mb
FROM d2d;

SELECT '' AS '';
SELECT 'Section 6: Top runtime CUDA APIs' AS section;
WITH api AS (
  SELECT 'ddp' AS method, s.value AS api_name, SUM(r.end - r.start) / 1e9 AS total_s
  FROM ddp.CUPTI_ACTIVITY_KIND_RUNTIME r
  JOIN ddp.StringIds s ON r.nameId = s.id
  GROUP BY s.value
  UNION ALL
  SELECT 'fsdp' AS method, s.value AS api_name, SUM(r.end - r.start) / 1e9 AS total_s
  FROM fsdp.CUPTI_ACTIVITY_KIND_RUNTIME r
  JOIN fsdp.StringIds s ON r.nameId = s.id
  GROUP BY s.value
), ranked AS (
  SELECT method, api_name, total_s,
         ROW_NUMBER() OVER (PARTITION BY method ORDER BY total_s DESC) AS rn
  FROM api
)
SELECT method, api_name, ROUND(total_s, 3) AS total_s
FROM ranked
WHERE rn <= 8
ORDER BY method, rn;

SELECT '' AS '';
SELECT 'Section 7: Top GPU kernels overall' AS section;
WITH kernels AS (
  SELECT 'ddp' AS method, s.value AS kernel_name, SUM(k.end - k.start) / 1e9 AS total_s
  FROM ddp.CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN ddp.StringIds s ON k.shortName = s.id
  GROUP BY s.value
  UNION ALL
  SELECT 'fsdp' AS method, s.value AS kernel_name, SUM(k.end - k.start) / 1e9 AS total_s
  FROM fsdp.CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN fsdp.StringIds s ON k.shortName = s.id
  GROUP BY s.value
), ranked AS (
  SELECT method, kernel_name, total_s,
         ROW_NUMBER() OVER (PARTITION BY method ORDER BY total_s DESC) AS rn
  FROM kernels
)
SELECT method, kernel_name, ROUND(total_s, 3) AS total_s
FROM ranked
WHERE rn <= 8
ORDER BY method, rn;

SELECT '' AS '';
SELECT 'Section 8: FT vs PEFT communication share by method' AS section;
WITH all_k AS (
  SELECT 'DDP FT' AS regime, SUM(end - start) / 1e9 AS total_kernel_s FROM ddp.CUPTI_ACTIVITY_KIND_KERNEL
  UNION ALL
  SELECT 'FSDP FT', SUM(end - start) / 1e9 FROM fsdp.CUPTI_ACTIVITY_KIND_KERNEL
  UNION ALL
  SELECT 'DDP PEFT', SUM(end - start) / 1e9 FROM ddp_peft.CUPTI_ACTIVITY_KIND_KERNEL
  UNION ALL
  SELECT 'FSDP PEFT', SUM(end - start) / 1e9 FROM fsdp_peft.CUPTI_ACTIVITY_KIND_KERNEL
), nccl_k AS (
  SELECT 'DDP FT' AS regime, SUM(k.end - k.start) / 1e9 AS nccl_kernel_s
  FROM ddp.CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN ddp.StringIds s ON k.shortName = s.id
  WHERE s.value LIKE 'ncclDevKernel%'
  UNION ALL
  SELECT 'FSDP FT', SUM(k.end - k.start) / 1e9
  FROM fsdp.CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN fsdp.StringIds s ON k.shortName = s.id
  WHERE s.value LIKE 'ncclDevKernel%'
  UNION ALL
  SELECT 'DDP PEFT', SUM(k.end - k.start) / 1e9
  FROM ddp_peft.CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN ddp_peft.StringIds s ON k.shortName = s.id
  WHERE s.value LIKE 'ncclDevKernel%'
  UNION ALL
  SELECT 'FSDP PEFT', SUM(k.end - k.start) / 1e9
  FROM fsdp_peft.CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN fsdp_peft.StringIds s ON k.shortName = s.id
  WHERE s.value LIKE 'ncclDevKernel%'
)
SELECT a.regime,
       ROUND(a.total_kernel_s, 3) AS total_kernel_s,
       ROUND(n.nccl_kernel_s, 3) AS nccl_kernel_s,
       ROUND(100.0 * n.nccl_kernel_s / a.total_kernel_s, 2) AS nccl_pct_of_total_kernel_time
FROM all_k a
JOIN nccl_k n USING(regime)
ORDER BY CASE a.regime
  WHEN 'DDP FT' THEN 1
  WHEN 'FSDP FT' THEN 2
  WHEN 'DDP PEFT' THEN 3
  WHEN 'FSDP PEFT' THEN 4
END;

SELECT '' AS '';
SELECT 'Section 9: FT vs PEFT device-to-device memcpy volume' AS section;
WITH d2d AS (
  SELECT 'DDP FT' AS regime,
         SUM(bytes) / 1024.0 / 1024.0 AS total_mb,
         COUNT(*) AS copies
  FROM ddp.CUPTI_ACTIVITY_KIND_MEMCPY m
  JOIN ddp.ENUM_CUDA_MEM_KIND sk ON m.srcKind = sk.id
  JOIN ddp.ENUM_CUDA_MEM_KIND dk ON m.dstKind = dk.id
  WHERE sk.label = 'Device' AND dk.label = 'Device'
  UNION ALL
  SELECT 'FSDP FT',
         SUM(bytes) / 1024.0 / 1024.0,
         COUNT(*)
  FROM fsdp.CUPTI_ACTIVITY_KIND_MEMCPY m
  JOIN fsdp.ENUM_CUDA_MEM_KIND sk ON m.srcKind = sk.id
  JOIN fsdp.ENUM_CUDA_MEM_KIND dk ON m.dstKind = dk.id
  WHERE sk.label = 'Device' AND dk.label = 'Device'
  UNION ALL
  SELECT 'DDP PEFT',
         SUM(bytes) / 1024.0 / 1024.0,
         COUNT(*)
  FROM ddp_peft.CUPTI_ACTIVITY_KIND_MEMCPY m
  JOIN ddp_peft.ENUM_CUDA_MEM_KIND sk ON m.srcKind = sk.id
  JOIN ddp_peft.ENUM_CUDA_MEM_KIND dk ON m.dstKind = dk.id
  WHERE sk.label = 'Device' AND dk.label = 'Device'
  UNION ALL
  SELECT 'FSDP PEFT',
         SUM(bytes) / 1024.0 / 1024.0,
         COUNT(*)
  FROM fsdp_peft.CUPTI_ACTIVITY_KIND_MEMCPY m
  JOIN fsdp_peft.ENUM_CUDA_MEM_KIND sk ON m.srcKind = sk.id
  JOIN fsdp_peft.ENUM_CUDA_MEM_KIND dk ON m.dstKind = dk.id
  WHERE sk.label = 'Device' AND dk.label = 'Device'
)
SELECT regime,
       ROUND(total_mb / 1024.0, 1) AS total_gb,
       copies
FROM d2d
ORDER BY CASE regime
  WHEN 'DDP FT' THEN 1
  WHEN 'FSDP FT' THEN 2
  WHEN 'DDP PEFT' THEN 3
  WHEN 'FSDP PEFT' THEN 4
END;

SELECT '' AS '';
SELECT 'Section 10: FT vs PEFT collective pattern summary' AS section;
WITH nccl AS (
  SELECT 'DDP FT' AS regime, s.value AS kernel_name, SUM(k.end - k.start) / 1e9 AS total_s
  FROM ddp.CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN ddp.StringIds s ON k.shortName = s.id
  WHERE s.value LIKE 'ncclDevKernel%'
  GROUP BY s.value
  UNION ALL
  SELECT 'FSDP FT', s.value, SUM(k.end - k.start) / 1e9
  FROM fsdp.CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN fsdp.StringIds s ON k.shortName = s.id
  WHERE s.value LIKE 'ncclDevKernel%'
  GROUP BY s.value
  UNION ALL
  SELECT 'DDP PEFT', s.value, SUM(k.end - k.start) / 1e9
  FROM ddp_peft.CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN ddp_peft.StringIds s ON k.shortName = s.id
  WHERE s.value LIKE 'ncclDevKernel%'
  GROUP BY s.value
  UNION ALL
  SELECT 'FSDP PEFT', s.value, SUM(k.end - k.start) / 1e9
  FROM fsdp_peft.CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN fsdp_peft.StringIds s ON k.shortName = s.id
  WHERE s.value LIKE 'ncclDevKernel%'
  GROUP BY s.value
), ranked AS (
  SELECT regime,
         kernel_name,
         total_s,
         ROW_NUMBER() OVER (PARTITION BY regime ORDER BY total_s DESC) AS rn
  FROM nccl
)
SELECT regime, kernel_name, ROUND(total_s, 3) AS total_s
FROM ranked
WHERE rn <= 3
ORDER BY CASE regime
  WHEN 'DDP FT' THEN 1
  WHEN 'FSDP FT' THEN 2
  WHEN 'DDP PEFT' THEN 3
  WHEN 'FSDP PEFT' THEN 4
END, rn;
