WITH joined AS (
    SELECT
        i.event_id,
        i.user_id,
        i.content_id,
        i.entry_surface,
        i.device,
        i.watch_minutes,
        i.value_score,
        i.retained_7d,
        u.subscription_tier,
        u.region,
        c.genre,
        c.language,
        c.quality_score
    FROM interactions i
    JOIN users u USING (user_id)
    JOIN content c USING (content_id)
)
SELECT
    genre,
    subscription_tier,
    COUNT(*) AS events,
    ROUND(AVG(value_score), 3) AS avg_value_score,
    ROUND(AVG(watch_minutes), 2) AS avg_watch_minutes,
    ROUND(AVG(retained_7d), 3) AS retention_rate
FROM joined
GROUP BY genre, subscription_tier
HAVING COUNT(*) >= 50
ORDER BY avg_value_score DESC, events DESC
LIMIT 20;
