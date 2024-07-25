1. **`api/libs/rsa.py`**

   - 缓存私钥数据以减少文件系统的访问频率。

   ```
   python
   复制代码
   cache_key = 'tenant_privkey:{hash}'.format(hash=hashlib.sha3_256(filepath.encode()).hexdigest())
   private_key = redis_client.get(cache_key)
   ```

2. **`api/core/app/features/rate_limiting/rate_limit.py`**

   - 缓存速率限制信息，包括当前活跃请求数和最大允许请求数。

   ```
   python
   复制代码
   redis_client.hset(self.active_requests_key, request_id, str(time.time()))
   ```

3. **`api/services/dataset_service.py`**

   - （代码内容过多，未全部展示）通常用于缓存数据集相关的同步状态和索引状态。

4. **`api/core/rag/datasource/vdb/milvus/milvus_vector.py`**

   - 使用Redis锁来管理矢量索引的创建和存在状态。

   ```
   python
   复制代码
   lock_name = 'vector_indexing_lock_{}'.format(self._collection_name)
   with redis_client.lock(lock_name, timeout=20):
       collection_exist_cache_key = 'vector_indexing_{}'.format(self._collection_name)
       if redis_client.get(collection_exist_cache_key):
           return
   ```

5. **`api/tasks/sync_website_document_indexing_task.py`**

   - 缓存文档同步状态以避免重复处理。

   ```
   python
   复制代码
   sync_indexing_cache_key = 'document_{}_is_sync'.format(document_id)
   ```

1. **`api/core/helper/tool_provider_cache.py`**

   - **缓存数据类型**：工具提供者的凭据。

   - 代码片段

     ：

     ```
     python
     复制代码
     redis_client.setex(self.cache_key, 86400, json.dumps(credentials))
     ```

2. **`api/tasks/annotation/batch_import_annotations_task.py`**

   - **缓存数据类型**：批量导入注解的状态和错误信息。

   - 代码片段

     ：

     ```
     python
     复制代码
     indexing_cache_key = 'app_annotation_batch_import_{}'.format(str(job_id))
     redis_client.setex(indexing_cache_key, 600, 'completed')
     ```

1. **`api/core/rag/datasource/vdb/analyticdb/analyticdb_vector.py`**

   - **缓存数据类型**：矢量索引状态。

   - 代码片段

     ：

     ```
     python
     复制代码
     cache_key = f"vector_indexing_{self._collection_name}"
     lock_name = f"{cache_key}_lock"
     with redis_client.lock(lock_name, timeout=20):
         collection_exist_cache_key = f"vector_indexing_{self._collection_name}"
         if redis_client.get(collection_exist_cache_key):
             return
         redis_client.set(collection_exist_cache_key, 1, ex=3600)
     ```

2. **`api/services/account_service.py`**

   - **缓存数据类型**：用户登录状态、密码重置令牌、邀请令牌。

   - 代码片段

     ：

     ```
     python
     复制代码
     redis_client.set(_get_login_cache_key(account_id=account.id, token=token), '1', ex=int(exp.total_seconds()))
     ```

     ```
     python
     复制代码
     redis_client.setex(
         cls._get_invitation_token_key(token),
         expiryHours * 60 * 60,
         json.dumps(invitation_data)
     )
     ```

3. **`api/core/rag/datasource/vdb/opensearch/opensearch_vector.py`**

   - **缓存数据类型**：矢量索引状态。

   - 代码片段

     ：

     ```
     python
     复制代码
     lock_name = f'vector_indexing_lock_{self._collection_name.lower()}'
     with redis_client.lock(lock_name, timeout=20):
         collection_exist_cache_key = f'vector_indexing_{self._collection_name.lower()}'
         if redis_client.get(collection_exist_cache_key):
             return
         redis_client.set(collection_exist_cache_key, 1, ex=3600)
     ```